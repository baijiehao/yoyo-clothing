import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# ==================== 模型重建（和原来一模一样） ====================
def create_model(num_classes=3):
    model = models.efficientnet_b0(weights='DEFAULT')  # 用默认预训练权重
    # 冻结所有层（和原来一样，只训head）
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model

# ==================== 加载你的模型 ====================
@st.cache_resource  # 只加载一次，节省内存
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"模型文件不存在：{model_path}")
        return None
    
    model = create_model(num_classes=3)
    checkpoint = torch.load(model_path, map_location='cpu')  # 用CPU避免显存问题
    
    # 兼容你原来保存的方式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', ['不喜欢', '喜欢', '一般般'])
    else:
        model.load_state_dict(checkpoint)
        class_names = ['不喜欢', '喜欢', '一般般']
    
    model.eval()
    return model, class_names

# ==================== 图片预处理 ====================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="优优宝宝衣服喜好预测", page_icon="👗")
st.title("👗优优宝宝衣服喜好预测器")
st.markdown("### 上传一张衣服图片，小天才的模型来告诉你：优优对这件衣服：不喜欢 / 喜欢 / 一般般")
st.caption("模型准确率约48.7%（基于现有小数据集），仅供参考～")

# 填写你的模型路径
MODEL_PATH = "clothing_model_efficientnet.pth"  # ←←← 修改成你的实际文件名！！！

model, class_names = load_model(MODEL_PATH)

if model is None:
    st.stop()

# 上传图片
uploaded_file = st.file_uploader("上传衣服图片（jpg/png）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示原图
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="上传的图片", use_column_width=True)
    
    # 预处理
    input_tensor = transform(image).unsqueeze(0)  # 加batch维度
    
    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(outputs, dim=1).item()
        confidence = probs[pred_idx].item()
    
    pred_label = class_names[pred_idx]
    
    # 显示结果
    st.markdown(f"## 🏆 预测结果：**{pred_label}**")
    st.markdown(f"### 置信度：{confidence*100:.1f}%")
    
    # 显示所有概率
    st.markdown("#### 详细概率")
    for i, name in enumerate(class_names):
        prob = probs[i].item() * 100
        st.progress(prob / 100)
        st.caption(f"{name}: {prob:.1f}%")

st.markdown("---")
st.caption("模型基于训练数据，仅供参考～ 如果想更准，多收集点图片再训练哦！")