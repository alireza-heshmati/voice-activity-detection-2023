import math
import timeit
import copy

from tqdm import tqdm

import torch
from models.utils import wav_label_to_frame_label_pyannote

def F1_Score(TP, FP, TN, FN):
    """calcuale F1-Score criteria

    Arguments
    ---------
    TP : int
        True positive for calculating F1-Score criteria
        
    FP : int
        False positive for calculating F1-Score criteria

    TN : int
        True negative for calculating F1-Score criteria

    FN : int
        False positive for calculating F1-Score criteria

    Returns
    -------
    output : float
        F1_Score

    """
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return 2 * Precision * Recall / (Precision + Recall + 1e-10)

# calcuale MCC criteria
def MCC(TP, FP, TN, FN):
    """calcuale F1-Score criteria

    Arguments
    ---------
    TP : int
        True positive for calculating MCC criteria
        
    FP : int
        False positive for calculating MCC criteria

    TN : int
        True negative for calculating MCC criteria

    FN : int
        False positive for calculating MCC criteria

    Returns
    -------
    output : float
        MCC
    """
    return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-10)



def evaluate_epoch(model, data_loader, loss_fn, target_fn, frame_pyannote_fn, device):
    """Evaluate model with loss, F1-Score and MCC

    Arguments
    ---------
    model : class
        Model for validating
        
    data_loader : class
        Test or validation data loader

    loss_fn : function
        Loss function
        

    target_fn : function
        Create framed label according to Pyannote

    frame_pyannote_fn: function
        Define the number and the length of frames according to Pyannote model
        

    device : str
        CPU or GPU


    Returns
    -------
    loss : float
        Loss
        
    f1 : float
        F1_score
        
    mcc : float
        MCC
    """
    
    model.eval()

    loss = 0
    TP = 0 # pred 1, actual 1
    FP = 0 # pred 1, actual 0
    TN = 0 # pred 0, actual 0
    FN = 0 # pred 0, actual 1
    counter = 0

    with torch.no_grad():  
        for data, _, frm_targ in tqdm(data_loader):
            num_frame, len_frame = frame_pyannote_fn(data.shape[-1])
            
            output = model(data.to(device)).cpu()
            del data

            output = torch.repeat_interleave(output,len_frame, dim=-2).squeeze()
            output_t, output  = wav_label_to_frame_label_pyannote(output, output.shape[-1]//320, 320)

            frm_targ = frm_targ[:,:,None]
            if output_t.shape[1] < frm_targ.shape[1]:
                frm_targ = frm_targ[:,:output_t.shape[1],:]

            loss += loss_fn(output, frm_targ)

            ind_pred = output_t == 1
            ind_target =  frm_targ == 1
            
            # Calculate TP, FP, FN, TN
            TP += len(frm_targ[ind_pred * ind_target])
            FP += len(frm_targ[ind_pred * ~ind_target])
            FN += len(frm_targ[~ind_pred * ind_target])
            TN += len(frm_targ[~ind_pred * ~ind_target])

            counter += 1

    f1 = F1_Score(TP, FP, TN, FN)
    mcc = MCC(TP, FP, TN, FN)
    loss = loss.cpu().item() / counter

    return round(loss, 5), round(f1, 3), round(mcc, 3)
                

def train_epoch(model, dataloader, optimizer, loss_fn, target_fn, device, step_show):
    """train each epoch

    Arguments
    ---------
    model : class
        Model for training
        
    data_loader : class
        Training data loader

    optimizer : function
        Training optimizer
        
    loss_fn : function
        Loss function

    target_fn : function
        Create framed label according to Pyannote

    device : str
        CPU or GPU

    step_show : int
        Number of batches to reduce learning rate and show training results


    Returns
    -------
    total_loss : float
        Train loss for the epoch
    """
    
    model.train()
    
    total_loss = 0
    counter = 0
    ex_counter = 0
    section = 1
    loss_section = 0

    start = timeit.default_timer()
    for data, target,_ in tqdm(dataloader):
        target = target_fn(target).to(device)

        output = model(data.to(device))
        loss = loss_fn(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        counter += 1
        
        # change the lr at defined sections of each epoch
        if counter % step_show == 0:
            finish = timeit.default_timer()

            lr = optimizer.param_groups[0]['lr']
            l = (total_loss.detach().cpu().item() - loss_section) / (counter - ex_counter)
            print(f"Section {section}. lr: {lr:.2f}, Loss: {l:.4f}, Time (Min): {round((finish - start) / 60, 3)}")

            loss_section = total_loss.detach().cpu().item()
            ex_counter = counter
            for g in optimizer.param_groups:
                if g['lr'] > 0.1:
                    g['lr'] = round(g['lr'] - 0.01, 2)
            section += 1
            start = finish

    if section == 1 :
        finish = timeit.default_timer()

        lr = optimizer.param_groups[0]['lr']
        l = total_loss.detach().cpu().item() / counter
        print(f"Section {section}. lr: {lr:.2f}, Loss: {l:.4f}, Time (Min): {round((finish - start) / 60, 3)}")

        loss_section = total_loss.detach().cpu().item()
        for g in optimizer.param_groups:
            if g['lr'] > 0.1:
                g['lr'] = round(g['lr'] - 0.01, 2)
        start = finish
        
    total_loss = total_loss.detach().cpu().item() / counter
    print(f" Train Loss: {total_loss:.5f}")

    return total_loss



# run the training and evaluation.
def run(model,
        train_loader,
        validation_loader,
        test_loader,
        optimizer,
        loss_fn,
        target_fn,
        frame_pyannote_fn,
        device,
        save_model_path,
        step_show,
        n_epoch,
        ):
    """train each epoch

    Arguments
    ---------
    model : class
        Model for training
        
    train_loader : class
        Training data loader
        
    validation_loader : class
        Validation data loader
        
    test_loader : class
        Test data loader

    optimizer : function
        Training optimizer
        
    loss_fn : function
        Loss function

    target_fn : function
        Create framed label according to Pyannote

    frame_pyannote_fn: function
        Define the number and the length of frames according to Pyannote model

    device : str
        CPU or GPU
        
    save_model_path : str
        Path for saving model parameters

    step_show : int
        Number of batches to reduce learning rate and show training results

    n_epoch : str
        Number of epoches

    Returns
    -------
    train_losses : float
        Train losses
        
    val_losses : float
        Validation losses
        
    val_fscores : float
        Validation f1_score
        
    val_mccs : float
        Validation mcc
        
    test_loss : float
        Test loss
        
    test_fscore : float
        Test f1_score
        
    test_mcc : float
        Test mcc
        
    """
    
    best_loss = 1e10
    train_losses = []
    train_lrs = []
    val_losses = []
    val_fscores = []
    val_mccs = []

    for epoch in range(n_epoch):
        start = timeit.default_timer()
        train_lrs.append(optimizer.param_groups[0]['lr'])
        print('\n',f"--- start epoch {epoch+1} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, target_fn, device, step_show)
        train_losses.append(train_loss)
        
        val_loss, val_fscore, val_mcc = evaluate_epoch(model, validation_loader, loss_fn, target_fn, frame_pyannote_fn, device)
        val_losses.append(val_loss)
        val_fscores.append(val_fscore)
        val_mccs.append(val_mcc)

        finish = timeit.default_timer()
        print(f"Val_Loss: {val_loss:.4f}, Val_F1score: {val_fscore:.3f}, Val_MCC: {val_mcc:.3f}, Epoch_Time (min): {round((finish - start) / 60, 3)}")

        # save best model
        if best_loss > val_loss:
            best_loss = val_loss         
            best_fscore = val_fscore
            best_mcc = val_mcc
            best_epoch = epoch + 1
            best_train_lr = train_lrs[-1]
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), save_model_path) 

    print(f"\nTop validation accuracy. Epoch: {best_epoch}, lr: {best_train_lr},Best_loss: {best_loss:.4f}, Best_Fscore: {best_fscore:.3f}, Best_MCC: {best_mcc:.3f}")

    #test evaluation
    test_loss, test_fscore, test_mcc = evaluate_epoch(model, test_loader, loss_fn, target_fn,  frame_pyannote_fn, device)
    print(f"\nTest accuracy on Best model. Test_loss: {test_loss:.4f}, Test_Fscore: {test_fscore:.3f}, Test_MCC: {test_mcc:.3f}")
    return train_losses, val_losses, val_fscores, val_mccs, test_loss, test_fscore, test_mcc



