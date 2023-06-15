Epistemic markers in the scientific discourse   
======================

The central role of such epistemic concepts as theory, explanation, model, or mechanism is rarely questioned in philosophy of science. Yet, what is their actual use in the practice of science? 

In this philosophy of science project, we deploy text-mining methods to investigate the usage of 61 epistemic notions in a corpus of full-text articles from the biological and biomedical sciences (N=73,771). The influence of disciplinary context is also examined by splitting the corpus into sub-disciplinary clusters. 

The results reveal the intricate semantic networks that these concepts actually form in the scientific discourse, not always following our intuitions, at least in some parts of science.

The research is published in: Malaterre, Christophe & Martin Léonard. 2023. "Epistemic markers in the scientific discourse", Philosophy of Science.

The subject being incredibly rich, both in terms of data and of philosophical inquiry, many questions remain unexplored. The whole toolset is thus made available to anyone interested in pursuing further investigations. Detailed steps to reproduce the specific results that were published are included. 


Project structure and workflow
------------------------------

We organised the repository to ensure both the reproducibility of the results presented in the publication, and the availability of the toolset to pursue further investigations.

The "run_step..." scripts correspond to the main steps described in the methods section of the article:
- run_step_1_preprocess.py: Data extraction and preprocessing. Generation of the docterm matrix used in step 3.
- run_step_2_cooccurrences.py: Generation of the cooccurrence matrix. This step can be omitted.
- run_step_3_topics_clusters: Topic modeling and clustering.
- run_step_4_lexcounts: Lexicon word counts.
- run_step_5_results: Export the final results (average word frequencies and correlation tables).

These "run_step..." scripts make use of other core code used across the project and stored in the `/lib` directory. 

The code uses a DocModel class as a base data structure to represent the different documents forming the corpus. Storing the docmodels as pickles makes it possible to easily store the data for later use. A working corpus can be formed by filtering the docmodels to only keep those meeting certain criterion, like minimal text length or specific metadata values. For the published results, docmodel objects were created for all of the 119,340 XML files made available by BioMed, holding all the relevant data extracted from the raw XML. The docmodels were thereby filtered to retain research articles (to the extent possible), hence the 73,771 documents of the working corpus.

The basic workflow consists of using a generator to cycle through the docmodels, each time reading some data or performing some operation.
While this process is somewhat slow, it helps keep the memory footprint to a minimum by storing only the relevant data. Some useful generator functions can be found in `lib/utils/generators.py`

To run the analyses, simply edit the config file `config.py` and update the PATH variables to match your local configuration.
Scripts can then be run in order, execution details will be printed to the terminal.
The process can be quite lengthy and somme steps will require several hours if working with the whole corpus.

The code used in the various analyses is located in `lib/models`. 

Each type of analysis is represented by a class (acting as a type of model) holding all the relevant logic, which can be updated with the data from a DocModel generator.

Starting an analysis is simple as creating an instance of the class, using the init method to set parameters. In most cases, these classes have an update method which should be called once for each DocModel, passing the relevant data. Once done updating the model, the data can usually be exported as a DataFrame using the .as_df() method.

Note that the model itself can be stored as a pickle file, which is usually recommended. In certain cases, converting the results to a DataFrame can require a lot of memory and lead to errors. However, the models themselves are usually relatively lightweight, so saving them before exporting to a dataframe is a good precaution to avoid having to repeat the updating process, which can take a fair amount of time.

These analysis tools were designed to be flexible (most of the update methods work from a generic data type, like a list of strings or a list of Tag-like objects) and could easily be used with a different corpus.


Available analyses
------------------

This is an overview of the different tools available to conduct a series of analyses on the docmodels. See the documentation within the modules themselves for more details.

#### TagCounts

Location: `lib/models/tagcounts.py` 
Class name: `TagCountsModel`

Makes it possible to count a variety of tags following POS tagging of the text content stored in the docmodels. 
For example, count the frequency of each word across the corpus, while also counting the number of its instances as noun, verb, adjective etc.

#### Docterm

Location: `lib/models/docterm.py` 
Class name: `DocTermModel`

Collects the data and provides the functionalities needed to build a docterm matrix. 
The model itself is relatively lightweight and the vocabulary can be filtered using the results from a TagCountsModel before exporting the matrix, making it easier to avoid memory issues.

#### LDA Topic Modeling

Location: `lib/models/ldatopics.py` 
Class name: `LdaModel`

Wrapper for SKlearn's LatentDirichletAllocation providing a few useful shortcuts.

#### Cooccurrences

Location: `lib/models/coocs.py` 
Class name: `CoocsModel`

Counts the cooccurrence between each pair of terms within a specified vocabulary (for instance: explanation, theory, model, mechanism, understanding, prediction), within given window (defaults to +/- 5 terms centered on the target term). 
For each pair of terms, also logs the reference id (typically a doc-paragraph identifier) of each excerpt in which they are found cooccurring.


Notes on reproducibility
------------------

Due to copyright, the BioMed articles are not stored in the present repository, but should be downloaded directly from BioMed. The list of all the articles we used can be found in `srs/data/legacy/legacy_ids.json`, together with the list of documents and terms of the docterm matrix `srs/data/legacy/legacy_docterm_labels.json`.

Because some minor corrections were made to the text extraction process late in the project, we devised a legacy mode to ensure reproducibility. To do so, LEGACY_MODE should be set to True in `config.py`
Indeed, after the working corpus was built based on the 150 words abstract and 2000 words text thresholds, and after the analyses were done, we found out some non-text content -mostly urls- had filtered through (mostly by examining text excerpts in which specific cooccurrences were present). Corrections were made, but resulted in a handful of documents (a few dozens) being slightly (most by 10 words or less) below the 150 words abstract / 2000 words text thresholds. 
If using legacy mode, the working corpus will be built from a list of doc ids, in order to include these documents, thereby making it possible to exactly reproduce the published results.
Otherwise, the process will be run from scratch, and thoses files will be excluded from the working corpus (since they fall below the thresholds). The LDA results used to cluster the documents into disciplinary clusters might differ slightly, although the overall picture and conclusions should not be affected.

The code used to compute the docterm matrix was refactored before publishing, to improve performance and reusability. While this had no effect on the values themselves, it did change the ordering of the rows and columns within the matrix. This reordering does not change the overall results. Yet, to reproduce the exact same output from the LDA topic modeling, a manual reordering based on the original configuration is required as in `srs/data/legacy/legacy_docterm_labels.json`.


Supplementary Information
------------------

This folder includes a file with SI Table S1 mentionned in the publication (frequency of semantic fields per cluster) as well as a data-for-graph file


