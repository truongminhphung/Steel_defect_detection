
# Steel Defect Detection

## Description
- The project applies Deep Learning techniques to autimatically detect defects in the steel production
- It's helpful for early recognising any errors happening in the steel production to deal with it as soon as possible
- I have used Resnet18 and Unet in this task

## Demo
HomePage

![HomePage](https://raw.githubusercontent.com/truongminhphung/Steel_defect_detection/master/demo/homePage.png)

Demo
![Demo](https://raw.githubusercontent.com/truongminhphung/Steel_defect_detection/master/demo/demo.gif)

The Steel is normal
![Normal](https://raw.githubusercontent.com/truongminhphung/Steel_defect_detection/master/demo/normal.png)

The steel is in error 3
![Error](https://raw.githubusercontent.com/truongminhphung/Steel_defect_detection/master/demo/error3.png)

The steel is in error 3 and error 4 
![Error](https://raw.githubusercontent.com/truongminhphung/Steel_defect_detection/master/demo/error_4.png)

The steel is in error 3 and error 4
![Error](https://raw.githubusercontent.com/truongminhphung/Steel_defect_detection/master/demo/error4_1.png)

## Dataset
- This dataset contains 5 outputs according to 4 defects in steels and the steel has no defect.
- Kaggle dataset: https://www.kaggle.com/c/
severstal-steel-defect-detection/data

## Trained model
- Link download the trained model: https://drive.google.com/file/d/1ofi3O4LPY2SjMF4v176PsmXQ2kw898-G/view?usp=sharing

## Installation
install inpendences:
```bash
pip install -r requirements.txt
```

## Run model
- Download the trained model and placed it in steel_defect_detection folder
- Run app.py