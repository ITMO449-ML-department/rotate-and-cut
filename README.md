# ROTATE-AND-CUT
#### Крутит, вырезает ряды, возвращает массив bounding box'ов(4 точки, составляющие прямоугольник, описывающий ряд на картинке)  

<details><summary>Инструкция</summary>
<p>

```python3

import bb_getter.bb_getter2 as bbox_getter  


NAME = "PATH_TO_IMAGE"
bboxes = bbox_getter.get_bb(NAME, save_path=save_path, verbose=0)  

```

</p>
</details>

<details><summary>Requirements</summary>
<p>

* python 3
* pip install -r requirements.txt

</p>
</details>

By Кирилл Лалаянц и Владислав Дюжев  
For Geoscan with LOVE