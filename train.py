import os
import sys
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_loader, valid_x, valid_y, model, args):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    best_acc = 0
    last_step = 0
    for epoch in range(1, args.epochs+1):
        for step, (b_x, b_y) in enumerate(train_loader):
            output = model(b_x)  # cnn output
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % args.test_interval == 0:
                test_output = model(valid_x)
                pred_y = torch.max(test_output, 1)[1].data
                accuracy = float((pred_y == valid_y.data).sum()) / float(valid_y.size(0))
                if accuracy > best_acc:
                    best_acc = accuracy
                    last_step = step
                    if args.save_best:
                        save(model, args.save_dir, 'best', step)
                else:
                    if step - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)


def save(model, save_dir, save_prefix, step):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, step)
    torch.save(model.state_dict(), save_path)
