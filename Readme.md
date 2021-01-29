# 3D Detection's visualization in pointclouds 
Visualize 3d detection results on colored pointclouds
## Example
![image](https://github.com/XYunaaa/3dDetectionVis/blob/master/result/000000.png)

![image](https://github.com/XYunaaa/3dDetectionVis/blob/master/result/000080.png)
## Install
    main package : open3d 0.9.0 opencv-python 
## Usage
    python vis.py
    
    change AIMED_ID in vis.py to change the visual frames；
    
    Place your target detection results in data/ RES.json as follows:
    [{" class ":" vehicle ", "width" : 4.925102723965409, "height" : 1.73, "length" : 1.6730355786800375 ,
    "x" : 8.556364684791708, "y" : 33.343216162928236, "z" : 1.1788707710374808, 
    "rotationYaw" : 0.92, "rotationPitch" : 0, "rota TionRoll: 0, "" trackId frameIdx" : 3, "" : 0}],
    
    Currently, the target categories supported are: vehicle, ped, cyclist.
    Visual colors of the object categories can be modified or added by
    modifying the color_list in vis.py.
    
    Now vehicle is blue;ped is green;cyclist is red;
