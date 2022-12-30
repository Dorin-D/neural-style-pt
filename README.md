# Readme for the StyleTransFair team

## Style classifier ([notebook](https://github.com/Dorin-D/neural-style-pt/blob/master/stylefeatures.ipynb))

### Instructions
To use this notebook, you will have to set the variables below as you find them in the code. After doing so, you have to do is run all the cells and then you will have the classified styles.
Variables to set: 
  * style_folder : path to the good/bad styles (see format of folder in current repository, "neural-style-pt/GoodAndBadStyles/" )
  * folders : list of paths to folders which contain styles you want to classify
  * good_folders, bad_folders : two lists to paths where you want to copy good and bad style images, same list length as folders

Note: If you use this code on kaggle, you will want to download the results at the end; to do so, you have to archive the folders and you have to also set the following:
  * good_folder, bad_folder : path folders which you want to archive
  * good_archive, bad_archive : path where you want to save the archives

## Stylized dataset creation ([notebook](https://github.com/Dorin-D/neural-style-pt/blob/master/create-stylized-dataset.ipynb))

### Instructions
* First of all, you will have to create content folders and style folders following the format as described below, under "location" and "location_styles".
* Set the necessary variables as described below. 
  * n_classes : amount of classes to use in the stylized dataset creation
  * n_styles : amount of styles to use per class
  * n_samples : amount of images to create per style,class pair
  * output_location : output path of your stylized dataset; if you move this to a different folder than /kaggle/working/output, also change the last cell where you create the archive



(**NOTE**: You will end up with n_styles\*n_samples images per class (e.g. n_classes=4,n_styles=3,n_samples=40 will result in 120 images per class, for 4 classes). Content images will not be reused. Style images may be reused. Kaggle has a run limit of 12 hours, and it took me 10 hours to generate 480 images of resolution 512x512.
If an image is of a lower resolution than 512x512, then the original resolution will be kept. 
If there aren't n_classes with n_styles*n_samples images, you will be informed.)


* To create the stylized dataset, you need to set the following variables:
  * command, in function create_stylized_dataset : 
  * p_model_location : path where to download the neural style transfer model
  * location : path to your content dataset

### Additional information:
* Content folder format
  * data: folder containing all images
  * labels.csv : csv file with columns "FILENAME", "CATEGORY": FILENAME is the name of files inside data (e.g. "applauding_001.jpg"), CATEGORY is the class of the images
  * location_styles : path to your style dataset

* Style folder format:
  * data: folder containing all style images
  * labels.csv : csv file with columns "FILENAME", "STYLE": same principle as above



* Stylized folder format:
The output folder will contain a folder "data", same as with style/content dataset folders. The labels.csv will contain the following columns:
  * "ORIG_CATEGORY_FILENAME"
  * "CATEGORY"
  * "STYLE"
  * "ORIG_STYLE_FILENAME"
  * "FILENAME", where "FILENAME" is the original style filename merged with the content filename, with an underscore inbetween (e.g. style.jpg + content.jpg => style.jpg_content.jpg)
 
create_stylized_dataset : you can set in this function the parameters you want to use for the neural style transfer (e.g. whether to use color transfer, the learning rate, weight of styles etc, see commented lines for relevant variables)


