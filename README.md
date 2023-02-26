# ROTATE-AND-CUT
#### Крутит, вырезает ряды, возвращает массив bounding box'ов(наборы из 4 точек, составляющих прямоугольники, каждый из которых описывает ряд на картинке)  

<details><summary>Инструкция</summary>
<p>

```python3

import bb_getter.bb_getter2 as bbox_getter  
'''
name : str
    path to image
save_path : str
    save path for plots
verbose : int
    0 - no info; 1 - results, important info; 2 - every step
intensity : {'keypoint', 'kmeansmask'}
    way of calculating intensity of a row ('keypoint' - by keypoint; 'kmeansmask' - by k-means mask)
smooth : bool 
    smooth intensity hist 
'''

bboxes = bbox_getter.get_bb(name, intensity = "keypoints", smooth = False, save_path=save_path, verbose=0)

```

</p>
</details>

<details><summary>Examples</summary>
<p>

В example.ipynb можно попробовать алгоритм на разных картинках из example_images.  

</p>
</details>

<details><summary>Requirements</summary>
<p>

* python3
* pip install -r requirements.txt

</p>
</details>

#### By Кирилл Лалаянц и Владислав Дюжев  
#### For Geoscan with LOVE