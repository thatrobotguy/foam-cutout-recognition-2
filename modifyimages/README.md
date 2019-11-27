
This script creates resized color data and gray data given color source data that is not necessary the correct dimensions.
You pass in 4 arguments.
1. The local root source directory or all of the manually classified images
2. The local output directory to place all of the resized images when they are created
3. The width of the resized image
4. The height of the resized image
Here is an example of running the script:
```
python3 resizegrayedge.py root_source/ output 700 200
```
The script will then generate the data.
