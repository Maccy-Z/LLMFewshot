import os
import torch
import pickle
from matplotlib import pyplot as plt
import shutil
import numpy as np


# Save into file. Automatically make a new folder for every new save.
class SaveHolder:
    def __init__(self, base_dir, nametag=None):
        dir_path = f'{base_dir}/saves'
        files = [f for f in os.listdir(dir_path) if os.path.isdir(f'{dir_path}/{f}')]
        existing_saves = sorted([int(f[5:]) for f in files if f.startswith("save")])  # format: save_{number}

        if existing_saves:
            save_no = existing_saves[-1] + 1
        else:
            save_no = 0

        self.save_dir = f'{base_dir}/saves/save_{save_no}'

        print("Making new save folder at: ")
        print(self.save_dir)
        os.mkdir(self.save_dir)

        shutil.copy2(f'{base_dir}/Fewshot/defaults.toml', f'{self.save_dir}/defaults.toml')
        shutil.copy2(f'{base_dir}/Fewshot/config.py', f'{self.save_dir}/config.py')
        if nametag is not None:
            with open(f'{self.save_dir}/tag.txt', 'w') as f:
                f.write(nametag)

        self.grads = []

    def save_model(self, model: torch.nn.Module, optim, epoch=0):
        # latest model
        torch.save({"model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict()}, f'{self.save_dir}/model.pt')
        # Archive model
        if epoch % 10 == 0:
            torch.save({"model_state_dict": model.state_dict(),
                        "optim_state_dict": optim.state_dict()}, f'{self.save_dir}/model_{epoch}.pt')



    def save_history(self, hist_dict: dict):
        with open(f'{self.save_dir}/history.pkl', 'wb') as f:
            pickle.dump(hist_dict, f)

    def save_grads(self, grads: dict):
        self.grads.append(grads)
        with open(f'{self.save_dir}/grads.pkl', 'wb') as f:
            pickle.dump(self.grads, f)


class SaveLoader:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        print(f"Loading save at {save_dir}")
        # self.model = torch.load(f'{save_dir}/model.pt')

        with open(f'{save_dir}/history.pkl', "rb") as f:
            self.history = pickle.load(f)

    def plot_history(self):
        val_accs = self.history["val_accs"]
        val_accs = np.array(val_accs)
        val_accs = np.mean(val_accs, axis=-1)
        max_val_acc = np.max(val_accs)

        print(max_val_acc)

        train_accs = self.history["accs"]
        train_accs = np.array(train_accs)
        train_accs = np.array_split(train_accs, 100)
        train_accs = np.stack([np.mean(ary) for ary in train_accs])

        plt.plot(np.linspace(0, val_accs.shape[0], val_accs.shape[0]),
                 val_accs, label="Validation Acc")
        plt.plot(np.linspace(0, val_accs.shape[0], train_accs.shape[0]),
                 train_accs, label="Train Acc")

        plt.title(f'Maximum valdation accuracy: {max_val_acc}')
        plt.ylabel("Accuracy %")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(f'{self.save_dir}/accs.png')
        plt.show()

    def plot_grads(self):
        with open(f'{self.save_dir}/grads.pkl', "rb") as f:
            grad_list = pickle.load(f)

        grad_mean, grad_std = {}, {}
        for epoch_grads in grad_list:
            for name, abs_grad in epoch_grads.items():
                mean, std = torch.std_mean(abs_grad, unbiased=False)
                mean, std = mean.item(), std.item()
                if "norm" in name:
                    continue
                if name not in grad_mean:
                    grad_mean[name] = [mean]
                    grad_std[name] = [std]
                else:

                    grad_mean[name].append(mean), grad_std[name].append(std)

        layer_names = list(grad_std.keys())
        for i, name in enumerate(layer_names):
            layer_names[i] = layer_names[i].replace(".weight", ".w")
            layer_names[i] = layer_names[i].replace(".bias", ".b")
            layer_names[i] = layer_names[i].replace("weight_model.w_", "")
            layer_names[i] = layer_names[i].replace("d2v_model", "d2v")
            layer_names[i] = layer_names[i].replace("res_modules.", "")
            layer_names[i] = layer_names[i].replace("lin_", "")

        grad_means, grad_stds = list(grad_mean.values()), list(grad_std.values())
        grad_means, grad_stds = np.array(grad_means).T, np.array(grad_stds).T

        for epoch, (mean, std) in enumerate(zip(grad_means, grad_stds)):
            plt.plot(mean)
            plt.xticks(range(len(layer_names)), layer_names, rotation="vertical")
            plt.subplots_adjust(bottom=0.3)
            plt.title(f'{epoch = }')
            plt.ylabel("abs gradient magnitude")
            plt.show()


if __name__ == "__main__":
    import re

    BASEDIR = "."
    SAVE_NO = 4

    def sort_key(filename):
        match = re.compile(r'(\d+)').search(filename)
        if match:
            return int(match.group(1))
        else:
            return filename

    saves = [f for f in os.listdir(f'{BASEDIR}/saves') if os.path.isdir(f'{BASEDIR}/saves/{f}')]
    saves = [f for f in saves if f.startswith("save_")]
    saves = sorted(saves, key=sort_key)
    save_dir = f'{BASEDIR}/saves/{saves[SAVE_NO]}'
    print(save_dir)
    h = SaveLoader(save_dir=save_dir)
    h.plot_history()
    h.plot_grads()

