# Code and Data for CMP discovery

Source code and data used for the paper "Efficient discovery 
of frequently co-occurring mutations in a sequence database 
with matrix factorization".

[![DOI](https://zenodo.org/badge/936911464.svg)](https://doi.org/10.5281/zenodo.14969552)

## Description

We have developed a robust method for efficiently tracking multiple co-occurring 
mutations in a sequence database. Evolution often hinges on the interaction of 
several mutations to produce significant phenotypic changes that lead to the 
proliferation of a variant. However, identifying numerous simultaneous mutations 
across a vast database of sequences poses a significant computational challenge. 
Our approach leverages a non-negative matrix factorization (NMF) technique to 
automatically and efficiently pinpoint subsets of positions where co-mutations 
occur, appearing in a substantial number of sequences within the data set.

In short, this alogithm extracts co-mutational positions (CMPs) from DNA data.
The source code can be found in main.py. After launching the algorithm will:
* Unzip the raw data files if they are not already present in the directory
  as csv files
* Read the DNA and amino acid sequences from the csv data files.
* Perform NMF on the data set over the range of possible number of factors=[7,17]
* Save the resultant H matrices in two forms:
    * Raw H matrix data saved in csv.
    * H matrix heatmap for visualization saved as png.
* Extract a list of unique mutations from all the H matrix factors
    * Save results to Union_Extracted_Mutations.csv
* Apply a statistical analysis algorithm to filter out redundant/irrelevant mutations 
    * Save results to Extracted_Mutations_Without_Any_Statistically_Irrelvent_Subsets.csv
* Calculate DSIM values for NMF rank=[1,50]
    * Save results to H_rank_stats.csv
* Compute the powerset of 21 highest frequency single-point mutations.
    * Compute the frequency of each element in powerset accross data set.
    * Save results to Mutation_Frequency_Batch_X.csv where X is the file number
        * In order to prevent csvs from growing so large they can't be opened
          the size is capped at 300,000 rows. This section will save multiple csv files.        

DNA sequences were downloaded on September 4th 2022 from the National Center for 77
Biotechnology Information (NCBI) Virus SARS-COV-2 data-hub for complete protein 78
and nucleotide sequences [16]. We filtered for human host sequences that have a 79
complete surface glycoprotein and a corresponding nucleotide sequence. In order 
to fit on the github servers the data csv file was partitioned into five parts and compressed.

### Dependencies

This project has dependencies on the following python modules:

* collections
* pandas
* pathlib
* itertools
* csv
* seaborn
* matplotlib.pyplot
* nimfa
* scipy
* tqdm import tqdm
* numpy
* re
* zipfile
* os

It is recommended that you use a package manager such as Anaconda to help you install all the correct modules.

### Installing

You can either clone this repository into a git repo, or download
the project as a zip. The only difference is that if you download
as a zip file, then you will need to manually unzip the project.

### Executing program

* Open up a command terminal
* Navigate to the directory that contains the source files.
  (main.py, data zip files, this README, etc )
* On the command line run
    ```
    python main.py
    ```

## Authors

For authors see the CITATION.cff file.

## License

This project is licensed under the GNU General Public License v3.0 License - see the LICENSE.md file for details
