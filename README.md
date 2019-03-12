# Cross-Lingual Event Extraction( Trigger Identification and Argument Identification):

This repository hosts code for experiments on multilingual event detection mainly benchmarking the tasks of Trigger and Argument Identification/Classification in ACE 2005 dataset.

## Requirements:
The list of required packages are under requirements.txt. To install them all, run this command:

     pip install -r requirements.txt
     
In addition to that, preprocessing requires Stanford Core NLP Parser, for that you will need to download [CoreNLP](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip), add it to your classpath and start StanfordCoreNLPServer using the command:
    
     java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9001 -timeout 150000000
    
## How to Use the Code?
### 1. Preprocessing the dataset:
Under DataModule, run 
 
    python data_preprocessor.py 

which is adapted to work primarily on ACE05 and split it into train, dev and test using the doc_split and generates the data in BIO annotation scheme. The full data cannot be included as it is under license. Be sure, to change get_args.py script to comply with the desired options for processing:

- languages: choose the language that you would like to focus on separated by comma
- root-dir: change the root directory of the dataset
- data-ace-path: change the release of the data
- method: jmee for data like the [following](https://github.com/lx865712528/JMEE/blob/master/ace-05-splits/sample.json) or tagging to produce BIO annotation
- pre-dir: output directory for saving the preprocessed files

### 2. Running the dataset:
Now that you have the dataset in BIO Annotation scheme for which examples are provided in DatasModule/sample/, you can train and evaluate either monolingually or multilingually by running:

     python train.py
     
