# Object Detection
### YOLO Training

## 1. 도로 낙하물 
낙하물의 사전적 의미로는 위쪽에서 떨어져 내려오는 물건으로 되어 있다. 본 연구에서는 정상적인 도로위에서 있어서는 되지 않으며 사람 및 차량에 위해를 가할 수 있는 Object로 정의한다. 도로 낙하물은 특정한 특징을 가지지 않는다.

## 2. 도로 낙하물 탐지
도로 낙하물은 위에서 정의한 바와 같이 특정 형태를 가지지 않아 탐지하는데 있어 많은 어려움이 있으며 도로가 아닌 일반 인도에 있는 물건에 대해서는 도로 낙하물이라고 칭하지 않는다. 인도에 있는 Object는 탐지하지 않으며 도로위에서 차량등의 정상적인 요소를 제외한 Object를 탐지해야 된다는 의미로 귀결된다.
도로 낙하물은 빠르게 달리는 차안에서 탐지해야 되는 만큼 높은 처리 속도 및 정확도를 요한다. 따라서 기존에 사용하는 Feature 기반의 Detection은 사용하지 않으며 현재 학계에서 높은 정확도 및 처리속도로 검증된 YOLO Detection Model을 사용한다.

## 3. 도로 낙하물 탐지 방향
기존에는 객체중심의 Training Dataset 구성으로 객체 외 배경등의 요소를 제외하여 Dataset을 생성하여 주변 배경 및 환경에 변화와 관계없이 Object Detection을 하는 연구가 주를 이뤘다. 하지만, 본 연구에서는 Training Dataset 생성과정에서 객체 및 배경의 상관관계를 고려하여 특정 환경에서만의 Dataset을 구성하여 특정환경에서만 검출가능한 Object Detection을 구현한다.

<img src="https://user-images.githubusercontent.com/48273766/70598099-66ade580-1c2d-11ea-9ce8-f1f700aa880b.png">
Filter에 따른 Feature maps1

Reference: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition”, ‘IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE’, VOL. 37, NO. 9, SEPTEMBER 2015


<img src="https://user-images.githubusercontent.com/48273766/70598349-fd7aa200-1c2d-11ea-9b4e-9bb395e7589e.png">
Filter에 따른 Feature maps2

Reference: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, “Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition”, ‘IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE’, VOL. 37, NO. 9, SEPTEMBER 2015

## 4. Object Detection

(1) Color 기반의 Object Detection
- Color 기반의 표지판 인식 그림?
  
<img src="https://user-images.githubusercontent.com/58175211/70507724-479d4e00-1b70-11ea-9989-e6753468b232.png">
<img src="https://user-images.githubusercontent.com/58175211/70507728-4b30d500-1b70-11ea-9cf4-1a4bd39e0b8a.png">

- Color 분포가 동일한 표지판 그림??
  
<img src="https://user-images.githubusercontent.com/58175211/70507800-787d8300-1b70-11ea-8cfb-e89ea960aece.png">
<img src="https://user-images.githubusercontent.com/58175211/70507801-7b787380-1b70-11ea-871b-51f2678138cc.png">

위 그림?과 같이 Color 기반으로 Object Detection이 가능하지만 그림??과 같은 색의 분포가 비슷하거나 유사한 모양의 표지판을 구별하는 정확도가 낮은 문제점이 있다. 크기, 기상, 빛 변화등의 변수에 대응하기 어렵다.

(2) Feature Point 기반의 Object Detection

<img src="https://user-images.githubusercontent.com/58175211/70507942-bd091e80-1b70-11ea-8ddf-deb28a60ba79.png">

상기 그림?과 같이 특징점 기반의 Object Detection 많은 연구가 진행되었으나 낮은처리속도로 실시간 처리에 어렵다. GPU를 사용하고 SIFT외의 별도의 연산을 하지 않는다는 전제하에서도 약 6~7프레임 으로 고속주행하는 자동차에서 사용하기에는 제한된다. 특징점 기반의 탐지는 Color기반와 탐지보다는 변수들에 대한 성능저하가 상대적으로 적지만 낙하물 탐지에 적용가능한 수준의 성능을 기대할 수 없다.

## 4. Dataset 수집
- 배경이 도로인 낙하물data를 수집
- 도로 낙하물은 낙하물의 형태에 있어서는 정량적이지 않는 특징을 보이지만 도로라는 환경에 있어서는 동일한 특징을 보인다. 본 특징을 역이용하여 Preprocessing 단계에서 주위 환경과 관련없이 Dataset을 구성하는 것이 아닌 도로위에 있는 Object의 Data만을 취득하는 과정을 가진다.
- 도로 낙하물 Dataset 예시

<img src="https://user-images.githubusercontent.com/58175211/70530523-73332f00-1b96-11ea-867b-94b3ab0d7afa.png" width=400px>
<img src="https://user-images.githubusercontent.com/58175211/70530536-78907980-1b96-11ea-93cc-3467bccd884f.png" width=400px>
<img src="https://user-images.githubusercontent.com/58175211/70530551-80501e00-1b96-11ea-8557-aaa1d25bcdf6.png" width=400px>
<img src="https://user-images.githubusercontent.com/58175211/70531452-4bdd6180-1b98-11ea-8c0c-668244346dae.png" width=400px>
<img src="https://user-images.githubusercontent.com/58175211/70531806-0a00eb00-1b99-11ea-9841-93b9346c6c74.png" width=400px>
<img src="https://user-images.githubusercontent.com/58175211/70531172-b8a42c00-1b97-11ea-998a-13ed174acb7f.png" width=400px>
  
## 5. Dataset Preprocessing
[Preprocessing] https://github.com/AlexeyAB/Yolo_mark

- obj.data: 만드려는 클래스 갯수를 Classes변수에 저장
```
classes= 6
train  = data/train.txt
valid  = data/train.txt
names = data/obj.names
backup = backup/
```
- obj.names: 클래스의 이름을 저장
```
Obstacle
Car
Person
DelineatorPost
manhole
Line
```
- yolo-obj.cfg: 
1. batch와 subdivision을 설정 
2. region-classes변수에 원하는 클래스의 갯수 저장
3. convolutional-filters 변수에 (classes + 5) * 5 계산하여 저장
   
   [참조] https://github.com/AlexeyAB/Yolo_mark/blob/master/x64/Release/yolo-obj.cfg#L224

- img폴더안에 원하는 이미지를 넣고 /x64/Release/yolo_mark.cmd 실행

- Object의 크기만큼 Bounding Box를 표시  
<img src="https://user-images.githubusercontent.com/58175211/70502303-bfb14700-1b63-11ea-86c1-482ee96b4c31.png" width=400px>

## 6. Batch사이즈 별 Training 결과 확인
(1) Batch 4, Subdivision 2

<img src="https://user-images.githubusercontent.com/58175211/70504956-2b96ae00-1b6a-11ea-9bc5-b819788e1271.png">
<img src="https://user-images.githubusercontent.com/58175211/70505028-51bc4e00-1b6a-11ea-9bab-e7c51c5c74fb.png">
<img src="https://user-images.githubusercontent.com/58175211/70505031-541ea800-1b6a-11ea-86e3-b50d4aadd1cc.png">
<img src="https://user-images.githubusercontent.com/58175211/70505035-55e86b80-1b6a-11ea-9438-9ca1fed17de8.png">

- Training 결과

<img src="https://user-images.githubusercontent.com/58175211/70506742-600c6900-1b6e-11ea-852f-026ef6b6a804.png">  
<img src="https://user-images.githubusercontent.com/58175211/70505090-73b5d080-1b6a-11ea-8e08-d2adc6095b4b.png">


(2) Batch 8, Subdivision 4

<img src="https://user-images.githubusercontent.com/58175211/70505157-a8298c80-1b6a-11ea-946d-2637d52240f8.png">
<img src="https://user-images.githubusercontent.com/58175211/70505161-a9f35000-1b6a-11ea-9325-7615faa9e66d.png">
<img src="https://user-images.githubusercontent.com/58175211/70505168-abbd1380-1b6a-11ea-810f-66de2251bd83.png">
<img src="https://user-images.githubusercontent.com/58175211/70505173-ad86d700-1b6a-11ea-8ad4-e3db44103643.png">

- Training 결과

<img src="https://user-images.githubusercontent.com/58175211/70506798-7b777400-1b6e-11ea-83a2-435a6f0f9f0e.png">
<img src="https://user-images.githubusercontent.com/58175211/70505177-aeb80400-1b6a-11ea-89e4-bcaa06f4b0c1.png">


(3) Batch 64, Subdivision 4

<img src="https://user-images.githubusercontent.com/58175211/70505308-05bdd900-1b6b-11ea-9fc4-569a1446070e.png">
<img src="https://user-images.githubusercontent.com/58175211/70505314-08b8c980-1b6b-11ea-8ef0-b83771a6e876.png">
<img src="https://user-images.githubusercontent.com/58175211/70505331-0fdfd780-1b6b-11ea-882e-722eb33dadb1.png">
<img src="https://user-images.githubusercontent.com/58175211/70505335-11a99b00-1b6b-11ea-841c-1c137378e1c7.png">

-Training 결과

<img src="https://user-images.githubusercontent.com/58175211/70505337-13735e80-1b6b-11ea-908a-e3e7ab1fc6c5.png">

위와 같이 Batch, Subdivision 사이즈 변화에 따라 정확도가 증가하는 결과를 확인할 수 있다.

## 7. Data Augmentation
Preprocessing 단계에서 Dataset 공개된 바 없는 제한된 환경만의 Dataset을 취득하는 과정에서 데이터 생성의 한계가 있다. 본 문제를 해결하며 유동적인 환경에서도 효과적인 Object Detection을 위해 Augmentation을 통한 Training Dataset 확장을 적용한다. Augmentation은 Dataset 확장을 위해 흔하게 사용되는 기법으로 실제 환경에서 갑작스럽게 직면하는 변화에 대응할 수 있도록 한다. 

(1) Augmentation 기법 종류
-	Size up
-	Size down
-	Rotate left
-	Rotate right
-	Bright down
-	Blur
-	Gaussian Blur

(2) Augmentation 적용 예시

* 원본

<img src="https://user-images.githubusercontent.com/58175211/70503696-559aa100-1b67-11ea-871b-5375cb4fa59a.png" width=200px>


* Size up

<img src="https://user-images.githubusercontent.com/58175211/70503709-5e8b7280-1b67-11ea-8bb3-4a39c6219ebb.png">


* Size down

<img src="https://user-images.githubusercontent.com/58175211/70503713-60edcc80-1b67-11ea-8840-fbc20e2d941b.png">


* Rotate left

<img src="https://user-images.githubusercontent.com/58175211/70503731-6c40f800-1b67-11ea-8855-58de06a24b28.png" width=200px>


* Rotate right

<img src="https://user-images.githubusercontent.com/58175211/70503734-6d722500-1b67-11ea-998d-4e2af40065b3.png" width=200px>


* Bright down

<img src="https://user-images.githubusercontent.com/58175211/70503993-0b65ef80-1b68-11ea-983a-05963cad18b6.png" width=200px>


* Blur

<img src="https://user-images.githubusercontent.com/58175211/70504198-8f1fdc00-1b68-11ea-8a9d-fc3b30e8f95d.png" width=200px>


* Gaussian Blur

<img src="https://user-images.githubusercontent.com/58175211/70503743-719e4280-1b67-11ea-9070-f50daf4569a7.png" width=200px>

(3) Augmentation 실행 코드

```
from PIL import Image, ImageFilter, ImageEnhance
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

root_dir_path = 'C:/Users/okiju/Desktop/test' #target images directory
root_dir = os.listdir(root_dir_path)
print(root_dir)

path = 'C:/Users/okiju/Desktop/output' #output directory

for i in range(0, len(root_dir)):
 keyPath = os.path.join(root_dir_path, root_dir[i]) # keypath direct to root path
 print(keyPath)
 im = Image.open(keyPath)
 img = cv2.imread(keyPath)

# 블러
 blurImage = im.filter(ImageFilter.BLUR)
 blurImage.save('C:/Users/okiju/Desktop/output/blur_%s'%root_dir[i])

# 블러 - 가우시안
 GblurImage = im.filter(ImageFilter.GaussianBlur(2))
 GblurImage.save('C:/Users/okiju/Desktop/output/Gblur_%s'%root_dir[i])

# 회전
 rotateUPImage = im.rotate(20)
 rotateDOWNImage = im.rotate(-20)
 rotateUPImage.save('C:/Users/okiju/Desktop/output/rotateUP_%s'%root_dir[i])
 rotateDOWNImage.save('C:/Users/okiju/Desktop/output/rotateDOWN_%s'%root_dir[i])

# 밝기 - 밝게, 어둡
 bright = ImageEnhance.Brightness(im)
 brightUPImage = bright.enhance(1.5)
 brightDOWNImage = bright.enhance(0.5)
 brightUPImage.save('C:/Users/okiju/Desktop/output/brightUP_%s' % root_dir[i])
 brightDOWNImage.save('C:/Users/okiju/Desktop/output/brightDOWN_%s' % root_dir[i])

# 확대 UP, 축소 DOWN
 resizeUPImage = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
 resizeDOWNImage = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
 cv2.imwrite(os.path.join(path, 'resizeUP_%s'%root_dir[i]), resizeUPImage)
 cv2.imwrite(os.path.join(path, 'resizeDOWN_%s'%root_dir[i]), resizeDOWNImage)
```

## 8. Data Augmentation 후 Training 결과 확인
4와 같이 Dataset을 확장하고, Batch 사이즈 별 Training 후 결과를 확인한다.

(1) Batch 4, Subdivision 2

<img src="https://user-images.githubusercontent.com/58175211/70505883-6f8ab280-1b6c-11ea-9d61-f7faa32f2257.png">
<img src="https://user-images.githubusercontent.com/58175211/70505887-71547600-1b6c-11ea-9fce-09e24f93bac2.png">
<img src="https://user-images.githubusercontent.com/58175211/70505889-7285a300-1b6c-11ea-84c2-f897e68ba280.png">
<img src="https://user-images.githubusercontent.com/58175211/70505892-74e7fd00-1b6c-11ea-8a76-95189d4c650d.png">

- Training 결과

<img src="https://user-images.githubusercontent.com/58175211/70506925-b4174d80-1b6e-11ea-89e8-dcdfeb8f1b49.png">
<img src="https://user-images.githubusercontent.com/58175211/70506023-bed0e300-1b6c-11ea-92a2-9247f13f4efc.png">

(2) Batch 8, Subdivision 4

<img src="https://user-images.githubusercontent.com/58175211/70506070-d5773a00-1b6c-11ea-821a-9af45d1687f2.png">
<img src="https://user-images.githubusercontent.com/58175211/70506072-d740fd80-1b6c-11ea-95e2-baaf36024607.png">
<img src="https://user-images.githubusercontent.com/58175211/70506076-d8722a80-1b6c-11ea-9870-8758b0b0881c.png">
<img src="https://user-images.githubusercontent.com/58175211/70506078-dad48480-1b6c-11ea-9e26-7deb8df85706.png">

- Training 결과

<img src="https://user-images.githubusercontent.com/58175211/70506982-cf825880-1b6e-11ea-97b6-d55d5350bbe8.png">
<img src="https://user-images.githubusercontent.com/58175211/70506080-dc05b180-1b6c-11ea-9a0f-9a53239d4bd8.png">

(3) Batch 64, Subdivision 4

<img src="https://user-images.githubusercontent.com/58175211/70506202-15d6b800-1b6d-11ea-9e06-8bd1c0c057d4.png">
<img src="https://user-images.githubusercontent.com/58175211/70506206-17a07b80-1b6d-11ea-8b9d-1cbe87830e00.png"> 


<img src="https://user-images.githubusercontent.com/58175211/70506213-196a3f00-1b6d-11ea-9583-a105e1e533f9.png">
<img src="https://user-images.githubusercontent.com/58175211/70506215-1a02d580-1b6d-11ea-89a6-3e95d84a8451.png">

- Training 결과
  
<img src="https://user-images.githubusercontent.com/58175211/70506221-1cfdc600-1b6d-11ea-96b9-2d774609b5de.png">

위와 같이 Augmentation을 적용한 dataset 확장을 통한 정확도 향상이 되고 있으며 유동성을 높여 불균질한 낙하물을 잡는 정확도를 높일 수 있다.




## 10. Iteration 증가
<img src="https://user-images.githubusercontent.com/48273766/70602966-f0af7b80-1c38-11ea-80ed-44e155d92fd9.jpg">

Iteration의 증가에 따랄 정확도 개선이 가능하며 일정 시점이상에서 정확도가 증가하지 않는 부분은  Overfitting이 시작하는 구간이다. 해당 구간 근처의 weights를 이용하여 실험을 통한 Overfitting이 되지 않는 weights를 선별해야 한다.

- 최종 Training 결과
<img src="https://user-images.githubusercontent.com/48273766/70607090-2a848000-1c41-11ea-8350-9a1d40280a43.jpg">

## 9. Bounding Box 비교

<img src="https://user-images.githubusercontent.com/58175211/70508256-7667f400-1b71-11ea-98fe-70f266247496.png">

Batch 사이즈 변화 및 Aug 적용에 따라 Bounding Box의 정확도가 증가함을 보여준다.

## 10. Class 세분화
Class 세분화없이 낙하물만을 단독 Class로 학습하는 경우 낙하물만을 탐지하는 것이 아닌 다른 도로요소들도 탐지하는 경우가 발생했다. 본 문제 해결을 위해 도로위에서 잡힐 수 있는 요소들을 Class로 분류하여 최종적으로 탐지하고 하는 낙하물과 혼동되는 문제를 막는다.

(1) 도로 위의 맨홀을 낙하물로 잡는 문제 발생

<img src="https://user-images.githubusercontent.com/58175211/70508053-f8a3e880-1b70-11ea-85be-0ad626d7cd7e.png">
<img src="https://user-images.githubusercontent.com/58175211/70508059-fa6dac00-1b70-11ea-81d4-44d7d8fd02e5.png">

(2) 도로 위의 흰색 차선을 낙하물로 잡는 문제 발생

<img src="https://user-images.githubusercontent.com/58175211/70508066-fcd00600-1b70-11ea-9e05-c1f5b6ed8ff9.png">
<img src="https://user-images.githubusercontent.com/58175211/70508071-fe013300-1b70-11ea-9936-ca85fdd4bd42.png">

## 11. 거리추정

<img src="https://user-images.githubusercontent.com/48273766/70605054-18a0de00-1c3d-11ea-974a-8368be9154f2.png">
거리 추정 예시

<img src="https://user-images.githubusercontent.com/48273766/70604527-26a22f00-1c3c-11ea-8470-98776cfd02ae.png">
거리 추정 그래프

<img src="https://user-images.githubusercontent.com/48273766/70604832-ae883900-1c3c-11ea-8815-74a2a75baaff.png">
거리 추정식

Reference: 박래근, 윤혁진 외 8명, "YOLO기반의 ADAS 오작동 대응방안 연구",'대한전기학회 학술대회 논문집' , 2019.11, pp.188-190


```
float predict_dist = 0;
int obs_bot_dist;
obs_bot_dist = (show_img->rows - pt2.y)*(1080/show_img->rows);
printf("pt2.y: %d\n", pt2.y);
printf("obs_bot_dist:%d\n", obs_bot_dist);
predict_dist = (-962.38055/(obs_bot_dist - 534.5588)) - 0.57949466;
if (obs_bot_dist <= 530)
    printf("predict_dist:%f\n", predict_dist);
else
    printf("Out of Range\n");
```
거리 추정 코드

## 12. Postprocessing
- ROI 지정

낙하물이 도로 바닥을 제외한 공중에 있는 경우는 불가능하여 BoundingBox의 좌표가 영상 프레임 중 Image_heights/2 보다 작은 곳에서 잡히는 낙하물로 분류하지 않는다.

- Size 제한

영상 화면 전체를 채우는 이미지가 존재하는 것은 불가능하다. 실제 도로환경에서의 실험을 통해 탐지되는 낙하물의 객체의 Size 범위를 측정하여 해당범위를 벗어나는 경우 낙하물로 분류하지 않는다.

<img src="https://user-images.githubusercontent.com/48273766/70602562-10926f80-1c38-11ea-9687-61712e0d9b41.png">
사이즈 제한 수식

- 낙하물 Class를 제외한 Bounding Box 삭제

본 연구의 최종 목표인 낙하물을 제외한 Class의 결과를 표출하지 않는다.

- Postprocessing
```
if (rect_size>= rect_size_min && rect_size <= rect_size_max && rect_acc>=0 && pt2.y>=show_img->rows/2 && pt1.x>=0 && (strstr(labelstr, "Obstacle") != NULL))
{
    strcat(labelstr, " dist: ");
    char c_predict_dist[20];
    if (obs_bot_dist <= 530)
    {
        strcpy(c_predict_dist, to_string(predict_dist).c_str());
        strcat(labelstr, c_predict_dist);
        strcat(labelstr, "m");
    }
    else
    {
        strcpy(c_predict_dist, "Out of range");
        strcat(labelstr, c_predict_dist);
    }
    
    cv::rectangle(*show_img, pt1, pt2, color, width, 8, 0);
  
    cv::Scalar black_color = CV_RGB(0, 0, 0);
    cv::putText(*show_img, labelstr, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, black_color, 2 * font_size, CV_AA);
    
    src = *show_img;
    crop_src = src(Rect(left-10, top-15, rect_width, rect_height));
    char FilePath[600] = { 0, };
    sprintf_s(FilePath, "C:/users/PRG/Desktop/대회노트북백업/darknet-masterE/build/darknet/Obstacle/%02d-%02d-%02d %02d-%02d-%02d ,%f,%f.jpg", t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec, gps_rec.lat, gps_rec.log);
    save_mat_jpg(crop_src, FilePath);
    
    if(latti_interval>0.0001 && longi_interval>0.0001)
    {
        latti_temp = gps_rec.lat;
        longi_temp = gps_rec.log;
        FileSubmit(FilePath);
    }

}
 ```

## 13. 성능평가

- 실험 환경

103장의 Training Dataset외의 사진을 이용하여 실험을 진행한다.
103장내에서 발생하는 성능을 측정하여 결과를 도출한다.


-차량을 이용한 실험
<img src="https://user-images.githubusercontent.com/48273766/70606164-4a1aa900-1c3f-11ea-83ad-984866acd210.png">
ERP-42 플랫폼

<img src="https://user-images.githubusercontent.com/48273766/70606682-56533600-1c40-11ea-9cdb-ccd488998994.png">
ERP-42 플랫폼 스펙

Reference: Unmanned Solution Co.