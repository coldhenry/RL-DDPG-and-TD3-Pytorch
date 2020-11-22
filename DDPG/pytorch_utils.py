# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:25:57 2020

@author: coldhenry
"""
import torch
import shutil


def saveNetwork(model, directory, modelname):
    torch.save(model.state_dict(), directory + "{}.pth".format(modelname))
    print("Model has been saved...")


def loadNetwork(model, directory, modelname):
    model.load_state_dict(torch.load(directory + "{}.pth".format(modelname)))
    print("Model has been loaded...")


def checkpoint(epoch, model, optimizer):
    return {
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def save_checkpoint(state, checkpoint_dir, best_model_dir=None, is_best=False):
    f_path = checkpoint_dir + "checkpoint.pt"
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + "best_model.pt"
        shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"]


def DDPG_ckp(
    iterations,
    actor,
    actor_t,
    optimizer_actor,
    critic,
    critic_t,
    optimizer_critic,
    directory="",
):
    ckp_actor = checkpoint(iterations, actor, optimizer_actor)
    ckp_actor_t = checkpoint(iterations, actor_t, optimizer_actor)
    ckp_critic = checkpoint(iterations, critic, optimizer_critic)
    ckp_critic_t = checkpoint(iterations, critic_t, optimizer_critic)
    save_checkpoint(ckp_actor, directory)
    save_checkpoint(ckp_actor_t, directory)
    save_checkpoint(ckp_critic, directory)
    save_checkpoint(ckp_critic_t, directory)


def TD3_ckp(
    iterations,
    actor,
    actor_t,
    optimizer_actor,
    critic,
    critic_t,
    optimizer_critic,
    critic2,
    critic_t2,
    optimizer_critic2,
    directory="",
):
    ckp_actor = checkpoint(iterations, actor, optimizer_actor)
    ckp_actor_t = checkpoint(iterations, actor_t, optimizer_actor)
    ckp_critic = checkpoint(iterations, critic, optimizer_critic)
    ckp_critic_t = checkpoint(iterations, critic_t, optimizer_critic)
    ckp_critic2 = checkpoint(iterations, critic2, optimizer_critic2)
    ckp_critic_t2 = checkpoint(iterations, critic_t2, optimizer_critic2)
    save_checkpoint(ckp_actor, directory)
    save_checkpoint(ckp_actor_t, directory)
    save_checkpoint(ckp_critic, directory)
    save_checkpoint(ckp_critic_t, directory)
    save_checkpoint(ckp_critic2, directory)
    save_checkpoint(ckp_critic_t2, directory)
