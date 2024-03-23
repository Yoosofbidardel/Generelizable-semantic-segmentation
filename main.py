import torch
import torchvision
from torch import nn, optim
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import random
import wandb
import os
import tempfile
import shutil
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import prepare_data as pd
from torch.utils.data import DataLoader
from io import BytesIO
from base64 import b64encode
from typing import Dict
import unet_baseline
import siamese
import siamese_sgr
from IPython.display import display, HTML
        
#name = 'Siamese_contr_last'
name = 'Siamese_SGC_last'
#name = 'Siamese'
#name = 'unet'

#name = 'Siamese_sgr_first (3)'

class MeanIOU(object):
  def __init__(self, valid_classes, ignore_index=-1, max_label_value=None):
    self.valid_classes = valid_classes
    self.num_classes = len(self.valid_classes)
    self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
    self.ignore_index = ignore_index

    # Create mapping for label ids to eval ids (which are [0, num_classes))
    if max_label_value is None:
      max_label_value = max(self.valid_classes)
    self.lids2eval_ids = np.full(max_label_value+1, ignore_index, dtype=np.int32)
    for i in range(len(self.valid_classes)):
      self.lids2eval_ids[valid_classes[i]] = i
    self.lids2eval_ids = np.append(self.lids2eval_ids, ignore_index)
    #print(self.lids2eval_ids)


  def update(self, preds, labels):
    # assign ignore index also to other ignored items
    invalid_class_mask = np.logical_not(np.isin(labels, self.valid_classes))
    labels[invalid_class_mask] = len(self.lids2eval_ids)-1

    if np.any(np.logical_not(np.isin(preds, self.valid_classes))):
      present = np.unique(preds[np.logical_not(np.isin(preds, self.valid_classes))])
      raise ValueError("There are invalid classes in the prediction:", present)

    # Map prediction and labels to eval_ids
    seg_gt = self.lids2eval_ids[labels]
    seg_pred = self.lids2eval_ids[preds]

    non_ignore_index = seg_gt != self.ignore_index

    seg_gt = seg_gt[non_ignore_index]
    seg_pred = seg_pred[non_ignore_index]

    self.confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, self.num_classes)


  def result(self):
    pos = self.confusion_matrix.sum(1)
    res = self.confusion_matrix.sum(0)
    tp = np.diag(self.confusion_matrix)

    IU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IU = IU_array.mean()

    evaluation_results = {'meanIU': mean_IU, 'IU_array': IU_array, 'confusion_matrix': self.confusion_matrix}

    return evaluation_results

    def convert_confusion_matrix(confusion_matrix):
      cls_sum = confusion_matrix.sum(axis=1)
      confusion_matrix = confusion_matrix / cls_sum.reshape((-1, 1))
      return confusion_matrix

    print('evaluate segmentation:')
    meanIU = evaluation_results['meanIU']
    IU_array = evaluation_results['IU_array']
    confusion_matrix = convert_confusion_matrix(evaluation_results['confusion_matrix'])
    print('IOU per class:')
    for i in range(len(IU_array)):
      print('{}: '.format(i) + '%.5f' % IU_array[i])
    print('meanIOU: %.5f' % meanIU)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    import re
    confusion_matrix = re.sub('[\[\]]', '', np.array2string(confusion_matrix, separator='\t'))

    return {'meanIOU': float(mean_IU),
            'IOUs': list(IU_array)}

  def get_confusion_matrix(self, gt_label, pred_label, class_num):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param class_num: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * class_num + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((class_num, class_num))

    for i_label in range(class_num):
      for i_pred_label in range(class_num):
        cur_index = i_label * class_num + i_pred_label
        if cur_index < len(label_count):
          confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix
  
class Trainer:
    def __init__(self, model: nn.Module, ds_split: Dict[str,pd.SegmentationDataset]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.ds_split = ds_split

        #self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.lr = 0.001
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.scheduler = 0
        self.max_iter = 0

        #self.critereon = nn.MSELoss()
        self.critereon = nn.CrossEntropyLoss(ignore_index=255)
        self.critereon_sc = nn.CrossEntropyLoss()
        self.loss_contrastive = siamese_sgr.ContrastiveLoss()
        self.loss_sensitivity = siamese_sgr.SensitivtyGuidedLoss()
        self.sgr = siamese_sgr.SelfGuidedRandomization()

        self.best_loss = 1000

    def train_epoch(self, dl, alpha = 0, graphs = False):
        self.model.train()

        epoch_metrics = {
            "loss": [],
        }
        
        # Create a progress bar using TQDM
        sys.stdout.flush()
        with tqdm(total=len(dl.dataset), desc=f'Training') as pbar:
            for _, (inputs, truths) in enumerate(dl):
                self.optimizer.zero_grad()
                
                inputs = inputs.to(device=self.device, dtype=torch.float32)
                inputs.required_grad = True  # Fix for older PyTorch versions
                truths = truths.to(device=self.device, dtype=torch.long)

                # Run model on the inputs
                inputs_random = torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.3).forward(inputs)
                output = self.model.l1(inputs_random)
                #output = self.model(inputs_random)

                loss = 0

                if alpha is not 0:
                    inputs_random = torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.3).forward(inputs)
                    #output_2 = self.model(inputs_random)
                    output_2 = self.model.l1(inputs_random)

                    #output_new = self.sgr(output, output_2)
                    #output_2_new = self.sgr(output_2, output)
                    output_new = output
                    output_2_new = output_2

                    output_new = self.model.l2(output_new)
                    output_2_new = self.model.l2(output_2_new)

                    #loss = alpha * self.loss_contrastive(output_new, output_2_new)
                    #loss = alpha * self.loss_sensitivity(output, output_2, output_new, output_2_new)

                    output_new = self.model.l3(output_new)
                    output_2_new = self.model.l3(output_2_new)

                    loss = loss + alpha * self.loss_contrastive(output_new, output_2_new)
                    #loss = loss + alpha * self.loss_sensitivity(output, output_2, output_new, output_2_new)

                    loss = loss + (self.critereon(output_new, truths) + self.critereon(output_2_new, truths)) / 2
                    #loss = (self.critereon(output, truths) + self.critereon(output_2, truths)) / 2
                else:
                    output = self.model.l2(output)
                    output = self.model.l3(output)
                    loss = self.critereon(output, truths)

                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()

                #self.adjust_lr_poly(self.max_iter*epoch + list(inputs.shape)[0], 0.9)
                #self.scheduler.step()

                # Store the metrics of this step
                step_metrics = {
                    'loss': loss.detach(),
                }

                # log metrics to wandb
                if graphs:
                    wandb.log({"loss_train": loss})

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])

                # Add to epoch's metrics
                for k,v in step_metrics.items():
                    epoch_metrics[k].append(v)

                #amount += 1
                #total_loss += step_metrics["loss"]

        sys.stdout.flush()

        # Print mean of metrics
        #total_loss /= amount

        # Return metrics
        # return {
        #     "loss": [total_loss],
        # }

        # Return metrics
        return epoch_metrics
    
    def val_epoch(self, dl, alpha = 0, graphs = False):
        self.model.eval()

        amount = 0
        total_loss = 0

        # Create a progress bar using TQDM
        sys.stdout.flush()
        with tqdm(total=len(dl.dataset), desc=f'Validation') as pbar:
            for _, (inputs, truths) in enumerate(dl):
                self.optimizer.zero_grad()

                inputs = inputs.to(device=self.device, dtype=torch.float32)
                inputs.required_grad = True  # Fix for older PyTorch versions
                truths = truths.to(device=self.device, dtype=torch.long)

                # Run model on the inputs
                inputs_random = torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.3).forward(inputs)
                output = self.model.l1(inputs_random)
                #output = self.model(inputs_random)

                loss = 0

                if alpha is not 0:
                    inputs_random = torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.3).forward(inputs)
                    #output_2 = self.model(inputs_random)
                    output_2 = self.model.l1(inputs_random)

                    #output_new = self.sgr(output, output_2)
                    #output_2_new = self.sgr(output_2, output)
                    output_new = output
                    output_2_new = output_2

                    output_new = self.model.l2(output_new)
                    output_2_new = self.model.l2(output_2_new)

                    #loss = alpha * self.loss_contrastive(output_new, output_2_new)
                    #loss = alpha * self.loss_sensitivity(output, output_2, output_new, output_2_new)

                    output_new = self.model.l3(output_new)
                    output_2_new = self.model.l3(output_2_new)

                    loss = loss + alpha * self.loss_contrastive(output_new, output_2_new)
                    #loss = loss + alpha * self.loss_sensitivity(output, output_2, output_new, output_2_new)

                    loss = loss + (self.critereon(output_new, truths) + self.critereon(output_2_new, truths)) / 2
                    #loss = (self.critereon(output, truths) + self.critereon(output_2, truths)) / 2
                else:
                    output = self.model.l2(output)
                    output = self.model.l3(output)
                    loss = self.critereon(output, truths)


                # Store the metrics of this step
                step_metrics = {
                    'loss': loss.detach(),
                }

                # Update the progress bar
                pbar.set_postfix(**step_metrics)
                pbar.update(list(inputs.shape)[0])

                amount += 1
                total_loss += step_metrics["loss"]

        sys.stdout.flush()

        # Print mean of metrics
        total_loss /= amount
        
        # log metrics to wandb  
        if graphs:
            wandb.log({"loss_test": total_loss})

        if self.best_loss > total_loss:
            print('Model saved')
            self.best_loss = total_loss
            save_model_linux(self.model, name)

        # Return metrics
        return {
            "loss": [total_loss],
        }

    def fit(self, epochs: int, batch_size:int, graphs = False):
        dl_train = DataLoader(self.ds_split["css"], batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(self.ds_split["css_val"], batch_size=batch_size, drop_last=True)
        
        #df_train = pd.DataFrame()
        #df_val = pd.DataFrame()

        self.max_iter = len(dl_train.dataset)/batch_size*epochs
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=self.max_iter, power=0.9)

        # Train the model for the provided amount of epochs
        for epoch in range(1, epochs+1):
            alpha = 0
            if epoch > 3:
               alpha = 10

            print(f'Epoch {epoch}')
            metrics_val = self.val_epoch(dl_val, alpha=alpha, graphs=graphs)
            #df_val = df_val.append(pd.DataFrame({'epoch': [epoch], **metrics_val}), ignore_index=True)

            metrics_train = self.train_epoch(dl_train, alpha=alpha, graphs=graphs)
            #df_train = df_train.append(pd.DataFrame({'epoch': [epoch for _ in range(len(metrics_train["loss"]))], **metrics_train}), ignore_index=True)

        # Return a dataframe that logs the training process. This can be exported to a CSV or plotted directly.
        #return df_train, df_val

            
def test_prediction(model: nn.Module, datasets):
    model.eval()
    model = model.cpu()

    template_table = '<table><thead><tr><th>Subset</th><th>Input sample</th><th>Output sample</th><th>Truth sample</th></tr></thead><tbody>{0}</tbody></table>'
    template_row = '<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td></tr>'
    template_img = '<img src="data:image/png;base64,{0}"/>'

    # Display a random sample of each split of the dataset
    rows = []
    for name in datasets:
        ds_sub = pd.ds_split[name]

        # Draw a random sample from the dataset so that we can convert it back to an image
        input, truth = random.choice(ds_sub)

        # Push through our network
        model = model.cpu()
        output = model(input.unsqueeze(0))

        input = TF.to_pil_image(input)
        truth = ds_sub.to_image(truth)
        output = ds_sub.to_image(ds_sub.masks_to_indices(output).squeeze(0))

        # Create a buffer to save each retrieved image into such that we can base64-encode it for diplay in our HTML table
        with BytesIO() as buffer_input, BytesIO() as buffer_truth, BytesIO() as buffer_output:
            input.save(buffer_input, format='png')
            output.save(buffer_output, format='png')
            truth.save(buffer_truth, format='png')

            # Store one row of the dataset
            images = [template_img.format(b64encode(b.getvalue()).decode('utf-8')) for b in (buffer_input, buffer_output, buffer_truth)]
            rows.append(template_row.format(name, *images))

    # Render HTML table
    table = template_table.format(''.join(rows))
    display(HTML(table))


def train_model(model: nn.Module, epochs=10, graphs = False):
    print("Testing training process...")

    if graphs:
        wandb.init(
            # set the wandb project where this run will be logged
            project="5aua0",
            
            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.001,
                "architecture": "Siamese-sgr-first-second",
                "dataset": "Cityscapes",
                "epochs": epochs,
            }
        )

    trainer = Trainer(model, pd.ds_split)
    trainer.fit(epochs=epochs, batch_size=25, graphs=graphs)

    if graphs:
        wandb.finish()

def test_model(model: nn.Module, datasets):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    critereon = nn.CrossEntropyLoss(ignore_index=255)

    #template_table = '<table><thead><tr><th>Dataset</th><th>Average</th><th>Flat</th><th>Construction</th><th>Object</th><th>Nature</th><th>Sky</th><th>Human</th><th>Vehicle</th></tr></thead><tbody>{0}</tbody></table>'
    #template_row = '<tr><td>{0}</td><td>{1}</td><td>{2}</td><td>{3}</td><td>{4}</td><td>{5}</td><td>{6}</td><td>{7}</td><td>{8}</td></tr>'
    class_ids = [i for i in range(19)]

    template_table = '<table><thead><tr><th>Dataset</th><th>Average</th>'
    template_row = '<tr><td>{0}</td><td>{1}</td>'

    classes_sorted = sorted(pd.classes, key=lambda x: x.trainId, reverse=False)
    for ids in class_ids:
        template_table = template_table + '<th>' + classes_sorted[ids].name + '</th>'
        template_row = template_row + '<td>{' + '{}'.format(ids+2) + '}</td>'

    template_table = template_table + '</tr></thead><tbody>{0}</tbody></table>'
    template_row = template_row + '</tr>'

    rows = []

    for ds in datasets:
        print("Testing process", ds)

        #ds_subset = torch.utils.data.Subset(ds_split[ds], [i for i in range(25)])
        ds_subset = pd.ds_split[ds]

        dl_test = DataLoader(ds_subset, batch_size=25, drop_last=True)
        
        conf_matrices = np.zeros((19,19))
        class_counts = np.zeros(19)
        miou = MeanIOU(valid_classes=np.arange(0,19),ignore_index=255)

        #total_loss = 0
        #results = []

        sys.stdout.flush()
        with tqdm(total=len(pd.ds_split[ds]), desc=f'Testing') as pbar:
            for _, (inputs, truths) in enumerate(dl_test):
                optimizer.zero_grad()

                inputs = inputs.to(device=device, dtype=torch.float32)
                #inputs.required_grad = True  # Fix for older PyTorch versions
                truths = truths.to(device=device, dtype=torch.long)

                # Run model on the inputs
                outputs = model(inputs)

                #loss = critereon(outputs, truths)
                #conf_matrices = np.add(conf_matrices, confusion_matrix(ds_split[ds].masks_to_indices(output).squeeze(0).flatten(), truths.flatten(), labels=class_ids))
                #results.append(intersect_and_union(ds_split[ds].masks_to_indices(output).squeeze(0), truths))
                
                outputs = np.array(pd.ds_split[ds].masks_to_indices(outputs).squeeze(0).cpu())
                truths = np.array(truths.cpu())

                for i in range(len(outputs)):
                    miou.update(preds=outputs[i], labels=truths[i])
                    
                #iou_cats = np.add(iou_cats, compute_iou_cat(ds_split[ds].masks_to_indices(output).squeeze(0), truths))
                
                #total_loss += loss.detach()

                pbar.update(list(inputs.shape)[0])

        sys.stdout.flush()

        #iou_class = compute_iou(conf_matrices, class_counts)

        #avg = np.divide(sum(iou_class), len(iou_class))
        #avg = np.divide(sum(np.multiply(iou_class, class_counts)), sum(class_counts))

        #iou_cats = np.divide(iou_cats, len(ds_split[ds]))
        #avg = np.divide(sum(iou_cats), len(iou_cats))

        #total_loss = total_loss/len(ds_split[ds])

        result = miou.result()
        meanIU = result['meanIU']
        IU_array = result['IU_array']

        IU_array = [round(x,2) for x in IU_array]
        rows.append(template_row.format(ds, round(meanIU*100, 2), *IU_array))

    # Render HTML table
    table = template_table.format(''.join(rows))
    #display(HTML(table))
    print(table)

def save_model(model: nn.Module, name: str):
    os.makedirs("models", exist_ok=True)

    temp = name
    count = 1
    while os.path.exists('models\\' + temp + '.pth'):
        temp = name + ' (' + str(count) + ')'
        count = count + 1
    torch.save(model.state_dict(), 'models\\' + temp + '.pth')

def save_model_linux(model: nn.Module, name: str):
    temp = name
    # count = 1
    # while os.path.exists(temp + '.pth'):
    #     temp = name + ' (' + str(count) + ')'
    #     count = count + 1
    torch.save(model.state_dict(), temp + '.pth')


if __name__ == "__main__":
    pd.prep_init()

    model = siamese_sgr.SiameseSGR()
    #model = unet_baseline.UnetModel()
    model.cuda()
    #train_model(model, epochs=50, graphs=True)
    #name = 'Siamese_sgr_first (3)'
    #save_model_linux(model, name)
    model.load_state_dict(torch.load(name + '.pth', map_location=torch.device('cuda')))

    #test_model(model, ['css_val', 'bdd', 'acdc', 'map'])
    test_prediction(model, ['css_val', 'bdd', 'acdc', 'map'])