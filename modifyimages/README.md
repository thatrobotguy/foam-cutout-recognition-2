
The script `resizegrayedge.py` takes in a directory containing classes of images denoted by a directory. It then creates 3 directories for `color`, `gray`, and `edge`. These directories each contain the image classes denoted by directories. The contents of the generated directories are simple. The `color` directory contains the original image, resized. The `gray` directory contains the images, resized, and then converted to grayscale. The `edge` directory contains the same images as the `gray` directory, but with an edge detector applied. All of th dimensions of these images are of the specified dimentions given in the command line arguments.

You pass in 4 arguments to the script.
1. The local root source directory or all of the manually classified images
2. The local output directory to place all of the resized images when they are created
3. The width of the resized image
4. The height of the resized image
Here is an example of running the script:
```
python3 resizegrayedge.py root_source/ output 700 200
```
The script will then generate the data.

The script `createtrainvaltest.py` takes in the directory containing the class definitions. This is the same as the previous script. This is how the script knows what datatypes there are.
Given the information on the image classes, this script will generate 3 directories. There will be a `train` directory, a `val` directory, and a `test` directory. Under the `train` and `val` directories, there will be a directory for each classification. If there are 6 classes in the root source image directory, there will be 6 folders underneath `train` and `val`. The `test` directory will have the same structure as `train` and `val`, but I am expecting this to change.

You pass 4 arguments to the script.
1. The local root source directory or all of the manually classified images
2. the local output directory that will contain `train`, `val`, and `test`
3. The percentage of all source images that we want to have in the `train` directory. This will be an integer from 0 to 100
4. The percentage of the images not used in `train` to be placed in the `val` directory This will be an integer from 0 to 100
Here is an exmaple of running the script:
```
python3 createtrainvaltest.py root_source/ trainvaltestdir 80 50
```
The script will then generate the data.
