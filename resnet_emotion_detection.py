import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets,transforms
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

transform = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    "test": transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
             }
root = Path.home() / "Downloads" / "Mimi_emotions_dataset"

train_data = datasets.ImageFolder(root / "train",transform=transform["train"])
test_data= datasets.ImageFolder(root / "val",transform=transform["test"])

train_dataloader = DataLoader(train_data,batch_size=8,shuffle=True,num_workers=0,drop_last=False)
test_dataloader = DataLoader(test_data,batch_size=30,shuffle=True,num_workers=0,drop_last=False)

model = models.resnet18(weights = ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512,3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
loss_fn= nn.CrossEntropyLoss()

for name,param in model.named_parameters():
    if 'fc' in name or 'layer4' in name:
        param.requires_grad=True
    else:
        param.requires_grad=False
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(),"lr":0.0001},
    {'params':model.fc.parameters(),"lr":0.001}
])
def train_one_epoch():
    model.train()
    total_loss=0
    for images,labels in train_dataloader:
        images,labels = images.to(device),labels.to(device)
        output= model(images)
        loss= loss_fn(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(train_dataloader)
def evaluate():
    model.eval()
    correct=0
    with torch.no_grad():
        for images,labels in test_dataloader:
            images,labels=images.to(device),labels.to(device)
            prediction=model(images).argmax(dim=1)
            correct+=(prediction==labels).sum().item()
        return correct/len(test_data)

num_epoch=10
best_accuracy=-1
best_epoch=0
for epoch in range(num_epoch):
    Average_loss= train_one_epoch()
    accuracy = evaluate()
    saved=""
    if accuracy>=best_accuracy:
        best_accuracy=accuracy
        best_epoch=epoch+1
        torch.save({"state_dict": model.state_dict(),
                    "classes": train_data.classes}, "mimi_best.pt")
        saved = "  <- saved"
    print(f"For epoch in range{epoch+1}, average loss: {Average_loss:.4}, accuracy: {accuracy:.2%}")
best = torch.load("mimi_best.pt", map_location=device)
model.load_state_dict(best["state_dict"])
def predict(img_path):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    x= transform["test"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    print(f"\nImage: {Path(img_path).name}")
    for cls, p in sorted(zip(train_data.classes, probs.tolist()), key=lambda t: -t[1]):
        print(f"  {cls:<8} {p:.1%}")
    print("Prediction:", train_data.classes[probs.argmax().item()])

#
# predict(Path.home() / "Downloads" / "mawa_test.jpeg")
def predict_frame(face_rgb):
    """Same maths as predict(), but takes a webcam crop instead of a file path."""
    img = Image.fromarray(face_rgb)
    x = transform["test"](img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = probs.argmax().item()
    return train_data.classes[idx], probs[idx].item()
def run_web_cam():
    model.eval()
    face_weights = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    face_detector = YOLO(face_weights)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # CAP_DSHOW uses microsoft directshow backend to access camera on windows
    while True:
        ok,frame = cap.read()
        #ok is boolean, if frame has been captured.
        #frame is numpy arrays of the pixels
        if not ok:
            break
        frame = cv2.flip(frame,1)
        results = face_detector(frame,conf=0.5,verbose=False)[0]
        for box in results.boxes.xyxy.cpu().numpy().astype(int):#so, the boxes in results and their coordinates xy get stored in cpu memory and converted into integers
            fx1,fy1,fx2,fy2= box
            fw=fx2-fx1
            pad = int(fw*.20)
            x1,y1= max(0,fx1-pad),max(0,fy1-pad)
            x2= min(frame.shape[1],fx2+pad)
            y2= min(frame.shape[0],fy2+pad)
            face_rgb = cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB) #cvtcolor= convert color, frame[y1:y2,x1:x2] extracts face from webcame
            label,conf= predict_frame(face_rgb)
            cv2.rectangle(frame, (fx1, fy1), (fx2,fy2), (0,255,0),2)
            cv2.putText(frame, f"{label} {conf:.0%}", (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Emotion detector - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

run_web_cam()


