import math
import timeit
import copy

from tqdm import tqdm

import torch

# calcuale F1-Score criteria
def F1_Score(TP, FP, TN, FN):
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    return 2 * Precision * Recall / (Precision + Recall + 1e-10)

# calcuale MCC criteria
def MCC(TP, FP, TN, FN):
    return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-10)


# Evaluate model with loss, F1-Score and MCC
def evaluate_epoch(model, data_loader, loss_fn, target_fn, device):
    model.eval()

    loss = 0
    TP = 0 # pred 1, actual 1
    FP = 0 # pred 1, actual 0
    TN = 0 # pred 0, actual 0
    FN = 0 # pred 0, actual 1
    counter = 0

    with torch.no_grad():  
        for data, target in tqdm(data_loader):
            target = target_fn(target)
            
            output = model(data.to(device)).cpu()
            loss += loss_fn(output, target)

            ind_pred = output > 0.5
            ind_target = target > 0.5
            
            # Calculate TP, FP, FN, TN
            TP += len(target[ind_pred * ind_target])
            FP += len(target[ind_pred * ~ind_target])
            FN += len(target[~ind_pred * ind_target])
            TN += len(target[~ind_pred * ~ind_target])

            del data, target,  ind_pred, ind_target
            counter += 1

    f1 = F1_Score(TP, FP, TN, FN)
    mcc = MCC(TP, FP, TN, FN)
    loss = loss.cpu().item() / counter

    return round(loss, 5), round(f1, 3), round(mcc, 3)



# train each epoch
def train_epoch(model, dataloader, optimizer, loss_fn, target_fn, device, step_show):
    model.train()
    
    total_loss = 0
    counter = 0
    ex_counter = 0
    section = 1
    loss_section = 0

    start = timeit.default_timer()
    for data, target in tqdm(dataloader):
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
        device,
        save_model_path,
        step_show,
        n_epoch,
        ):
    
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
        
        val_loss, val_fscore, val_mcc = evaluate_epoch(model, validation_loader, loss_fn, target_fn, device)
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
    test_loss, test_fscore, test_mcc = evaluate_epoch(model, test_loader, loss_fn, target_fn, device)
    print(f"\nTest accuracy on Best model. Test_loss: {test_loss:.4f}, Test_Fscore: {test_fscore:.3f}, Test_MCC: {test_mcc:.3f}")
    return train_losses, val_losses, val_fscores, val_mccs, test_loss, test_fscore, test_mcc



