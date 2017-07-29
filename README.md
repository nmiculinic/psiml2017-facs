# psiml2017-facs
Facial Activation Unit &amp; landmark detection for Petnica Machine learning summer school

# Datasets

## Cohn-Kanade

* 100 subject, 69% female to male (18-50 ages)
Frontal and 30 deg samples
* 6 emotions per person, FACS annotated
* 68 point landark per picture (*22 for frames since it's small videos; FACS is only last video frame)
* posed

640x490 image size; grayscale
~ 1000 FACS annotated, 22 000 Landmark annotated

## Jaffe images

* 10 female Japanese models, grayscale
Neutral + 6 basic emotions, 2-4 images per person
* ~200 images in total
* 256x256 grayscale images

## Pain crops

* 25 subjects in 225 directorie, ~50 000 files (48398)
* FACS for all images with intensity

## 10k 

* ~2200 pictures, 77 landmark positions
* 218x256 px
* 20 attributes information, like attractiveness, calmness, trustworthiness, etc.

## Utrecht 

* Posed paired images, 130 images of 75 people.
* One happy and one normal image per person
900x1200px images

## Kaggle -- Challenges in representation learning facial expression recognition challenge

* 48x48px facial grayscale images; already centered. Basic 6 emotions annotations.
* 28 709 examples


## Project plan/timeline

* Image pipeline for input data -- unified format
* getting image, dictionary attributes, 
```
{
        'image': image, 
        'FACS': [1, 3, 5], 
        'source': 'Jaffe', 
        'landmarks': [(0,3), (4,5)]
        'attributes': pd.Dataframe of attributes
}
```
Standardize images to faces only (Violet-Jones for example), preprocessed for background removal.

* Split train/CV/test set per person.
* Phase 1: single task only
* Phase 2: multi task only
* Perform error analysis
* Calcualte test set error and standard deviation
* Prepare presentation
* Showcase best/worst model mistakes/examples
* Draw some conclusions, make some metrics


# Lesson learned/project postmortem

* We only used Pain dataset, due to time constraints
* Despite good performance in training, and testing, on real webcam we observed unsatisfactory perfomance
* Keras fit_generator....
        * Multiprocessing is false by default --> Make true due to GIL
        * Default workers=1; no cool when generating new images is expensivish and you're running on 32 CPU workstation
        * Keras uses python multiprocessing queues, instead of TF ones, leadning to IO bottleneck between python->TF CPU->TF GPU data moving; We observed unsaturated GPU
* Need to clear TF graph after long hyperparameter search
```python
from keras import backend as K
K.clear_session()
```
* Assuming saving and restoring works...yeah not. We encountered a problem where keras happily saves the model, which he couldn't restore. There was issues with python closures. With this lesson learnt, we tested saving&restoring at training beginning
* Deep SELU worked the best among models; even with weak regularization. We tested Relu+BN and SELU outperformed it.
* Beginning BN helps accounting for initial input data weirdness
