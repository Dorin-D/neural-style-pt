import os
import pandas as pd
import numpy as np

"""
train/test set for each style
TODO: report update
TODO: add github to beginning of report
TODO: add gabriel to github
TODO: Upload code

"""


def choose_classes(labels, n_classes, n_train, n_test):
    """
    Chooses n_classes that contain at least n_train, n_test : train,test samples from the given labels dataframe.
    parameters:
        labels: pandas dataframe containing filenames, labels and train/test split
        n_classes: required number of classes
        n_train: required number of train samples
        n_test: required number of test samples
    returns:
        chosen_classes: an array of the names of the classes chosen    
    """
    #number of classes
    total_classes = labels['label'].nunique()
    #array of classes
    classes = labels.label.unique()

    assert total_classes >= n_classes, "n_classes must be smaller than the number of available classes. (choose_classes function)"

    #create random permutation of n_classes from all classes
    class_random_sampling = np.arange(total_classes)
    class_random_sampling = np.random.permutation(class_random_sampling)
    
    #count samples in each class,split
    samples_per_class = labels.groupby(['label','split'])['filename'].count().reset_index(name='count')
    chosen_classes = []
    for c in class_random_sampling:
        #get amount of samples per class for both splits
        samples_per_current_class = samples_per_class.loc[samples_per_class['label']==classes[c]]
        #get amount of samples per split
        train_samples_per_class = samples_per_current_class.loc[samples_per_current_class['split']=='train']['count'].reset_index(drop=True)[0]
        test_samples_per_class = samples_per_current_class.loc[samples_per_current_class['split']=='test']['count'].reset_index(drop=True)[0]
        if (train_samples_per_class < n_train) or (test_samples_per_class < n_test):
            #if not enough samples, we ignore this class and take the next one
            continue
        chosen_classes.append(classes[c])
        if len(chosen_classes) == n_classes:
            break

    return chosen_classes

def choose_styles(labels, n_styles):
    """
    Chooses n_styles randomly from the labels dataframe. 
    parameters:
        labels: pandas dataframe containing filenames and style labels of each file
        n_styles: number of styles to choose
    returns:
        chosen_styles: an array of the names of the styles chosen
    """
    #number of style classes
    total_classes = labels['label'].nunique()
    #array of classes
    classes = labels.label.unique()
    
    assert total_classes >= n_styles, "n_classes must be smaller than the number of available classes. (choose_styles function)"
    
    #create random permutation of n_classes from all classes
    class_random_sampling = np.arange(total_classes)
    class_random_sampling = np.random.permutation(class_random_sampling)

    chosen_styles = []
    for s in class_random_sampling:
        chosen_styles.append(classes[s])
        if chosen_styles == n_styles:
            break
    return chosen_styles
 
def create_stylized_dataset(location, location_styles, n_classes, n_styles, n_train, n_test, p_train, output_label):
    #any arguments for neural style transfer)
    """
    Given a domain, extracts (n_train, n_test) samples from n_classes. It assigns a dominant style to each class. For the class, 
        it assigns p_train*n_train (rounded down) samples to the dominant style, and (1-p_train)/(n_styles-1) (rounded down) samples to the non-dominant styles.
        Samples which are left out (due to rounding down) are assigned the dominant style. 
        Generates a .csv file which contains the filenames, the classes, the splits and the assigned style to them.
        TODO: Apply neural style transfer and create the actual datasets.
        TODO: keep content and style filenames
    parameters:
        location: the location of the domain to be used; format:
            location_folder
                >data
                    >>all images will be in this folder
                >labels.csv
                    >>this file will contain the columns: filename, label, split
                        >>filename: name of the file
                        >>label: name of the class
                        >>split: test or train; specifying the split of the sample
        location_styles: the location with the styles to be used; format:
            location_styles
                >data
                    >>all style images will be in this folder
                >labels.csv
                    >>this file will contain the columns: filename, label
                        >>filename: name of the style file
                        >>label: name of the style

        n_classes: amount of classes from the dataset to apply styles to
        n_styles: amount of styles to apply
            (n_classes, n_styles) should be equal?
        n_train: amount of train samples per class
        n_test: amount of test samples per class
            (n_train, n_test) should be equal?
        p_train: amount of bias in train set (p=0.9 => 90% of images will be of dominant class)
    output:
        tbd
    """
    data = os.path.join(location, "data")
    label_loc = os.path.join(location, "labels.csv")
    labels = pd.read_csv(label_loc)

    
    chosen_classes = choose_classes(labels, n_classes, n_train, n_test)
    assert len(chosen_classes) == n_classes, "Likely there aren't enough classes to have at least n_train,n_test samples (too many or too little classes were chosen)"

    style_label_loc = os.path.join(location_styles, "labels.csv")
    style_labels = pd.read_csv(style_label_loc)    
    chosen_styles = choose_styles(style_labels, n_styles)    
    assert len(chosen_styles) == n_styles, "Likely there aren't enough styles offered in the style label.csv file"

    #dominant ratio: the ratio of train samples in the dominant style
    dominant_ratio = p_train
    #non dominant ratio: the ratio of train samples in the non-dominant styles
    non_dominant_ratio = (1-p_train)/(n_styles-1)
    
    #amount of images in dominant and non-dominant styles
    images_dominant = int(n_train * dominant_ratio)
    images_non_dominant = int(n_train * non_dominant_ratio)
    
    #code doesn't work if ratios result in non-integer numbers; could be improved
    #update; code has been improved by adding leftover images to the dominant class
    #assert (images_dominant + (n_styles-1) * images_non_dominant) == n_train, "Proportions not worked out; pick values which result in whole numbers"

    #the assumption is that n_classes, n_styles are equal
    #pairs each class with a style which will be dominant
    dominant_style_class = []
    for c,s in zip(chosen_classes, chosen_styles):
        dominant_style_class.append([c,s])

    labels_wstyles = pd.DataFrame()


    for d_s_c in dominant_style_class:
        #select the class from the d_s_c pair
        c = d_s_c[0]
        #select samples from current class
        samples = labels[labels['label']==c].reset_index(drop=True)
        #select train samples for current class c
        train_samples = samples[samples['split']=='train'].reset_index(drop=True)
        #select test samples for current class c
        test_samples = samples[samples['split']=='test'].reset_index(drop=True)

        #get permutations for train,test samples
        random_train_permutation = np.random.permutation(np.arange(len(train_samples)))
        random_test_permutation = np.random.permutation(np.arange(len(test_samples)))


        #select first n_train, n_test samples from permutation
        chosen_trains = train_samples.iloc[random_train_permutation[0:n_train]][['filename', 'label', 'split']]
        chosen_tests = test_samples.iloc[random_test_permutation[0:n_test]][['filename', 'label', 'split']]
        #reset indices of chosen samples
        chosen_trains = chosen_trains.reset_index(drop=True)
        chosen_tests = chosen_tests.reset_index(drop=True)

        #get permutations for train,test samples
        chosen_train_permutation = np.random.permutation(np.arange(len(chosen_trains)))
        chosen_test_permutation = np.random.permutation(np.arange(len(chosen_tests)))

        #assign styles to train set
        for s in chosen_styles:
            if d_s_c[1]==s:
                #we have the dominant style
                #select first images_dominant to be of dominant styles, the remaining will be left for the other styles
                chosen_images = chosen_train_permutation[:images_dominant]
                chosen_train_permutation = chosen_train_permutation[images_dominant:]
                chosen_trains.loc[chosen_images, 'style'] = s

            else:
                #we have non dominant style
                #select first images_non_dominant to be of non dominant styles, the remaining will be left for the other styles
                chosen_images = chosen_train_permutation[:images_non_dominant]
                chosen_train_permutation = chosen_train_permutation[images_non_dominant:]
                chosen_trains.loc[chosen_images, 'style'] = s

        if len(chosen_train_permutation)>0:
            #if there's leftover images, assign them to dominant class
            chosen_trains.loc[chosen_train_permutation, 'style'] = d_s_c[1]

        #assign styles to test set
        for s in chosen_styles:
            #we have non dominant style
            #select first images_non_dominant to be of non dominant styles, the remaining will be left for the other styles
            chosen_images = chosen_test_permutation[:images_non_dominant]
            chosen_test_permutation = chosen_test_permutation[images_non_dominant:]
            chosen_tests.loc[chosen_images, 'style'] = s

        #assign leftover images to the last style in chosen_styles
        if len(chosen_test_permutation) > 0:
            chosen_tests.loc[chosen_test_permutation, 'style'] = chosen_styles[-1]

        labels_wstyles = pd.concat((labels_wstyles, chosen_trains), ignore_index=True)
        labels_wstyles = pd.concat((labels_wstyles, chosen_tests), ignore_index=True)


    #TODO: add neural style transfer
    #TODO: formalize output
    output_label = os.path.join(output_location, 'label.csv')
    labels_wstyles.to_csv(output_label)


    return 0


location = "/mnt/d0acc65e-8a85-435e-82fc-2ec7d0dbef67/University_And_Others/Paris-Saclay/Creation of an AI challenge challenge/Style-Trans-Fair/Datasets/MetaAlbum/PlantDiseases"
location_styles = "/mnt/d0acc65e-8a85-435e-82fc-2ec7d0dbef67/University_And_Others/Paris-Saclay/Creation of an AI challenge challenge/Style-Trans-Fair/Datasets/MetaAlbum/Styles"
n_classes = 3
n_styles = 3
n_train = 40
n_test = 40
p_train = 0.8
create_stylized_dataset(location, location_styles, n_classes, n_styles, n_train, n_test, p_train)









def choose_image_style(content, style, test_train_split):
    """
    neural style transferred image given style from test or train split
    """



























#previous work
"""
    DONE verify n_classes <= total classes
    DONE choose random classes 
        (choose random domain if domain==None, otherwise choose from the same domain)
    DONE >output as list classes
    DONE verify n_styles <= total styles
    DONE choose random styles
    DONE >output as list styles

    for each class:
        DONE verify n_train <= total train samples
        DONE verify n_test <= total test samples
    
        DONE select n_train random train samples 
        DONE select n_test random test samples
    
        #temporary assumption: n_classes == n_styles,
        #    otherwise multiple classes dominated by same style
        
        for the train set:
            DONE dominant_ratio = p
            DONE non_dominant_ratio = (1-p)/(s-1)
            matrix = []
            for i in n_styles:
                matrix.append([[non_dominant_ratio]*n_classes)])
            for i in range(n_styles OR n_classes, they're the same):
                matrix[i][i]=p
            
            DONE assert (int(n_train * p) + 
                int(n_train * non_dominant_ratio) + 
                int(n_train * non_dominant_ratio)) == n_train, 
                "Proportions not worked out; pick values which 
                result in whole numbers"

            assign style to each content image according to (class, p)
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    #do things with style[i], train[j] and matrix[i][j]
                    
                    output_size = max size between style, content_image
                    neural_style_transfer(style, train, parameters)
                    
                    


            for each class in classes:
                for each style in styles:   
                    

        for the test set:
            
"""




