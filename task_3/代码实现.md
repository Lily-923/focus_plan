
import torch.nn as nn  
class ESPCN(nn.Module):  
    def __init__(self, scale=4):  
        super().__init__()  
        self.body = nn.Sequential(  
            nn.Conv2d(3, 64, 5, padding=2), nn.Tanh(),  
            nn.Conv2d(64, 32, 3, padding=1), nn.Tanh(),  
            nn.Conv2d(32, 3*scale*scale, 3, padding=1),  
            nn.PixelShuffle(scale)  
        )  
  
    def forward(self, x):  
        return self.body(x)
  ```
#生成训练数据
```
  import cv2, random, os  
from pathlib import Path  
  
HR_DIR   = Path(r'D:\mosaic_sr\mosaic\demo_in\DIV2K_train_HR')  # 原高清  
OUT_DIR  = Path(r'D:\mosaic_sr\mosaic\data')                     # 训练数据  
OUT_DIR.mkdir(parents=True, exist_ok=True)  
SCALE = 4  
  
for img_path in HR_DIR.glob('*'):  
    hr = cv2.imread(str(img_path))  
    if hr is None:                # 防止空图  
        continue  
    # 生成 LR（马赛克）  
    h, w = hr.shape[:2]  
    lr   = cv2.resize(cv2.GaussianBlur(hr,(3,3),0), (w//SCALE, h//SCALE), interpolation=cv2.INTER_AREA)  
    # 保存成对  
    cv2.imwrite(str(OUT_DIR / f'{img_path.stem}_HR.png'), hr)  
    cv2.imwrite(str(OUT_DIR / f'{img_path.stem}_LR.png'), lr)  
    print(f"已生成：{img_path.name}")  
  
print("全部完成！")
```
#推理
```
import torch  
import cv2  
import os  
from pathlib import Path  
from model.net import ESPCN   # 你的模型类  
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
  
1. 创建并加载模型  
model = ESPCN().to(device)  
state = torch.load(r'D:\mosaic_sr\mosaic\weights\best.ckpt', map_location=device)  
model.load_state_dict(state, strict=False)  
model.eval()  
  
 2. 单张推理函数  
def process_image(image_path, output_path):  
    img = cv2.imread(str(image_path))  
    if img is None:  
        print('读图失败：', image_path)  
        return  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0  
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)  
  
    with torch.no_grad():  
        out = model(img).squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()  
    out = cv2.resize(out, (1000, 1000))  # ← 新增/修改  
    out = (out * 255).astype('uint8')  
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)  
  
3. 批量推理  
if __name__ == '__main__':  
    input_dir  = Path(r'D:\mosaic_sr\mosaic\JotangRecrument-main\JotangRecrument-main\ML\task_3\image_pairs\blurred')  
    output_dir = Path(r'D:\mosaic_sr\mosaic\results')  
    output_dir.mkdir(exist_ok=True)  
  
    for file in input_dir.glob('*.png'):  
        process_image(file, output_dir / file.name)  
  
    print('全部推理完成！结果保存在', output_dir)
```
#训练
```
import cv2  
import torch  
from torch.utils.data import Dataset, DataLoader  
import torch.nn as nn  
import pytorch_lightning as pl  
from pathlib import Path  
from model.net import ESPCN  # 确保正确导入 ESPCN 类  
  
#数据集类，读取数据  
class PairDataset(Dataset):  
    def __init__(self, root):  
        root = Path(root)  
        hr_files = sorted(root.glob('*_HR.png'))  
        lr_files = sorted(root.glob('*_LR.png'))  
        assert len(hr_files) == len(lr_files), f"HR 和 LR 文件数量不一致: HR({len(hr_files)}), LR({len(lr_files)})"  
        self.pairs = list(zip(lr_files, hr_files))  
        print(f"成功加载 {len(self.pairs)} 对训练样本")  
  
    def __len__(self):  
        return len(self.pairs)  
  
    def __getitem__(self, idx):  
        lr_path, hr_path = self.pairs[idx]  
        lr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)  
        hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)  
        if lr is None or hr is None:  
            raise FileNotFoundError(f"无法读取图片: {lr_path} 或 {hr_path}")  
        lr = cv2.resize(lr, (128, 128))  
        hr = cv2.resize(hr, (512, 512))  
        lr = torch.from_numpy(lr[:, :, ::-1] / 255.0).permute(2, 0, 1).float()  
        hr = torch.from_numpy(hr[:, :, ::-1] / 255.0).permute(2, 0, 1).float()  
        return lr, hr  
  
#模型  
class LitSR(pl.LightningModule):  
    def __init__(self):  
        super().__init__()  
        self.net = ESPCN(scale=4)  
  
    def forward(self, x):  
        return self.net(x)  
  
    def training_step(self, batch, idx):  
        lr, hr = batch  
        loss = nn.L1Loss()(self(lr), hr)  
        self.log('train_loss', loss)  
        return loss  
  
    def configure_optimizers(self):  
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)  
  
if __name__ == '__main__':  
    data_dir = Path(r'D:\mosaic_sr\mosaic\data')  # 确保路径正确  
  
    ds = PairDataset(data_dir)  
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=4)  
  
    model = LitSR()  
    trainer = pl.Trainer(max_epochs=200, accelerator='auto')  
    trainer.fit(model, dl)  
    torch.save(model.state_dict(), 'weights/best.ckpt')  
    print("训练完成！")
```
### 输出PSNR
```
import os  
import cv2  
import numpy as np  
from skimage.metrics import peak_signal_noise_ratio  
  
gt_dir   = r'D:\mosaic_sr\mosaic\JotangRecrument-main\JotangRecrument-main\ML\task_3\image_pairs\original'   # 真值（原高清）  
pred_dir = r'D:\mosaic_sr\mosaic\JotangRecrument-main\JotangRecrument-main\ML\task_3\image_pairs\blurred'    # 推理结果  
  
psnr_list = []  
  
for i in range(1, 41):  
    filename = f'{i}.png'  
    gt_path = os.path.join(gt_dir, filename)  
    pred_path = os.path.join(pred_dir, filename)  
  
    gt_img = cv2.imread(gt_path)  
    pred_img = cv2.imread(pred_path)  
  
    if gt_img is None or pred_img is None:  
        print(f"Missing file: {filename}")  
        continue  
  
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)  
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)  
  
    if gt_img.shape != pred_img.shape:  
        print(f"Size mismatch: {filename}")  
        continue  
  
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=255)  
  
    print(f"{filename}: PSNR={psnr:.4f}")  
  
    psnr_list.append(psnr)  
  
if psnr_list:  
    avg_psnr = np.mean(psnr_list)  
    print(f"\nAverage PSNR: {avg_psnr:.4f}")  
else:  
    print("No valid images evaluated.")
 ```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTQ2MTYyNDc5N119
-->