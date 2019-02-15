**Workflow of model**

**Stage1 : preprocess.py**

+-----------------+-----------------+-----------------+-----------------+
| Steps           | Objective       | Notes           |                 |
+=================+=================+=================+=================+
| Input           | [CSV            | [parameters]{.u | They are given  |
|                 | files]{.underli | nderline}       | as command line |
|                 | ne}             |                 | input           |
|                 |                 | -prefix for     |                 |
|                 | -Train.csv      | output files    | In case         |
|                 |                 |                 | categorical     |
|                 | -Test.csv       | -normalization  | features do not |
|                 |                 | method          | exist?          |
|                 | [Column(feature |                 |                 |
|                 | )               | -portion of     |                 |
|                 | information]{.u | train dataset   |                 |
|                 | nderline}       | to use for      |                 |
|                 |                 | validation(e.g. |                 |
|                 | -csv file       | 0.1)            |                 |
|                 | indicating      |                 |                 |
|                 | numeric         |                 |                 |
|                 | (continuous or  |                 |                 |
|                 | discrete)       |                 |                 |
|                 | features        |                 |                 |
|                 |                 |                 |                 |
|                 | -csv file       |                 |                 |
|                 | indicating      |                 |                 |
|                 | categorical     |                 |                 |
|                 | features        |                 |                 |
|                 |                 |                 |                 |
|                 | -csv file       |                 |                 |
|                 | indicating      |                 |                 |
|                 | label features  |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| Preprocessing   | -Number         |                 |                 |
|                 | encoding for    |                 |                 |
|                 | categorical     |                 |                 |
|                 | features        |                 |                 |
|                 |                 |                 |                 |
|                 | -Normalizing    |                 |                 |
|                 | for numeric     |                 |                 |
|                 | features        |                 |                 |
|                 |                 |                 |                 |
|                 | -Dividing train |                 |                 |
|                 | dataset into    |                 |                 |
|                 | train and       |                 |                 |
|                 | validation sets |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| Output          | [Pkl files for  | Pkl files for   |                 |
|                 | datasets]{.unde | column          |                 |
|                 | rline}          | information is  |                 |
|                 |                 | later required, |                 |
|                 | -\<prefix\>.tr. | when building   |                 |
|                 | pkl             | model           |                 |
|                 |                 |                 |                 |
|                 | -\<prefix\>.val |                 |                 |
|                 | .pkl            |                 |                 |
|                 |                 |                 |                 |
|                 | -\<prefix\>.tes |                 |                 |
|                 | t.pkl           |                 |                 |
|                 |                 |                 |                 |
|                 | [Pkl files for  |                 |                 |
|                 | column          |                 |                 |
|                 | information]{.u |                 |                 |
|                 | nderline}       |                 |                 |
|                 |                 |                 |                 |
|                 | -\<prefix\>.col |                 |                 |
|                 | s.pkl           |                 |                 |
+-----------------+-----------------+-----------------+-----------------+

**Stage2 : train.py**

+-----------------+-----------------+-----------------+-----------------+
| Steps           | Objective       | Notes           |                 |
+=================+=================+=================+=================+
| Input           | **For input and | [For model      | Trained model's |
|                 | output**        | saving and      | best params are |
|                 |                 | reload]{.underl | saved into a    |
|                 | [Pkl files for  | ine}            | file named as,  |
|                 | training]{.unde |                 |                 |
|                 | rline}          | -saveto: during | \<prefix\>.     |
|                 |                 | training, the   |                 |
|                 | -prefix: prefix | model writes    | \<saveto\>.     |
|                 | for saved       | its parameters  |                 |
|                 | dataset files   | into a file     | model\_best.pkl |
|                 | and column      | whenever the    | and             |
|                 | information     | model's         |                 |
|                 | file. It is     | validation      | \<prefix\>.     |
|                 | also used for   | score improves. |                 |
|                 | saving outputs  | *saveto* is a   | \<saveto\>.     |
|                 | to files. Used  | string to       |                 |
|                 | for loading the | include in the  | model\_options. |
|                 | following       | name of that    | pkl             |
|                 | files:          | file (in        |                 |
|                 |                 | addition to     |                 |
|                 | \**\<prefix\>.t | prefix)         |                 |
|                 | r.pkl*          |                 |                 |
|                 |                 | -isReload:      |                 |
|                 | \**\<prefix\>.v | Whether to load |                 |
|                 | al.pkl*         | existing model  |                 |
|                 |                 | parameters from |                 |
|                 | \**\<prefix\>.c | a file. Only    |                 |
|                 | ols.pkl         | possible when   |                 |
|                 | -- used for     | the model had   |                 |
|                 | building        | been training   |                 |
|                 | dataloader and  | previously      |                 |
|                 | model*          |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
|                 | **Model         | [training]{.und |                 |
|                 | hyperparameters | erline}         |                 |
|                 | **              |                 |                 |
|                 |                 | -batch size     |                 |
|                 | [Model          |                 |                 |
|                 | architecture]{. | -loss function  |                 |
|                 | underline}      |                 |                 |
|                 |                 | -optimizer      |                 |
|                 | -number of      |                 |                 |
|                 | hidden layers,  | -learning rate  |                 |
|                 | *an integer     |                 |                 |
|                 | between* *\[1,  | -maximum number |                 |
|                 | 4\]*            | of epochs       |                 |
|                 |                 |                 |                 |
|                 | -1^st^ hidden   | -validation     |                 |
|                 | layer size      | frequency       |                 |
|                 |                 |                 |                 |
|                 | -2^nd^ hidden   | -patience       |                 |
|                 | layer size      |                 |                 |
|                 |                 |                 |                 |
|                 | -3^rd^ hidden   |                 |                 |
|                 | layer size      |                 |                 |
|                 |                 |                 |                 |
|                 | -4^th^ hidden   |                 |                 |
|                 | layer size      |                 |                 |
|                 |                 |                 |                 |
|                 | -output layer   |                 |                 |
|                 | size            |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| Data loading    | Create          |                 |                 |
|                 | dataloader with |                 |                 |
|                 | train and       |                 |                 |
|                 | validation data |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| Model building  | -Define         |                 |                 |
|                 | embedding layer |                 |                 |
|                 | and             |                 |                 |
|                 | input/output    |                 |                 |
|                 | layer sizes     |                 |                 |
|                 |                 |                 |                 |
|                 | -Build model    |                 |                 |
|                 | using **model   |                 |                 |
|                 | architecture    |                 |                 |
|                 | parameters**.   |                 |                 |
|                 |                 |                 |                 |
|                 | -Save model     |                 |                 |
|                 | architecture    |                 |                 |
|                 | parameters to a |                 |                 |
|                 | file for later  |                 |                 |
|                 | use. File name: |                 |                 |
|                 | *"\<prefix\>.\< |                 |                 |
|                 | saveto\>.model\ |                 |                 |
|                 | _options.pkl"*  |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| Training        | -Train the      |                 |                 |
|                 | model using     |                 |                 |
|                 | given           |                 |                 |
|                 | **training      |                 |                 |
|                 | parameters**    |                 |                 |
|                 |                 |                 |                 |
|                 | \* use          |                 |                 |
|                 | *model.train()* |                 |                 |
|                 | function        |                 |                 |
|                 |                 |                 |                 |
|                 | \* saving of    |                 |                 |
|                 | trained         |                 |                 |
|                 | model-parameter |                 |                 |
|                 | s               |                 |                 |
|                 | is done inside  |                 |                 |
|                 | the             |                 |                 |
|                 | *model.train()* |                 |                 |
|                 | function.       |                 |                 |
|                 | (refer to       |                 |                 |
|                 | *model.train()* |                 |                 |
|                 | )               |                 |                 |
+-----------------+-----------------+-----------------+-----------------+
| output          | -\<prefix\>.\<s |                 |                 |
|                 | aveto\>.model\_ |                 |                 |
|                 | best.pkl        |                 |                 |
|                 |                 |                 |                 |
|                 | -\<prefix\>.\<s |                 |                 |
|                 | aveto\>.model\_ |                 |                 |
|                 | options.pkl     |                 |                 |
+-----------------+-----------------+-----------------+-----------------+

**Stage3 : test.py**

+-----------------------+-----------------------+-----------------------+
| Steps                 | Objective             | Notes                 |
+=======================+=======================+=======================+
| Input                 | -prefix for saved     |                       |
|                       | test dataset and      |                       |
|                       | model file            |                       |
|                       |                       |                       |
|                       | -name of trained      |                       |
|                       | model (loadfrom), for |                       |
|                       | saved model file      |                       |
|                       |                       |                       |
|                       | -batch size           |                       |
+-----------------------+-----------------------+-----------------------+
| Data loading          | Create dataloader     |                       |
|                       | with test data        |                       |
+-----------------------+-----------------------+-----------------------+
| Model building        | Create model with     |                       |
|                       | given model options.  |                       |
|                       | Then load the         |                       |
|                       | parameters for model  |                       |
|                       | saved.                |                       |
+-----------------------+-----------------------+-----------------------+
| Testing               | Test the model on     |                       |
|                       | dataset               |                       |
|                       |                       |                       |
|                       | \*use model.test      |                       |
|                       | function              |                       |
+-----------------------+-----------------------+-----------------------+
| Output                | Final score evaluated |                       |
|                       | and printed.          |                       |
+-----------------------+-----------------------+-----------------------+

**Inside model.py**

**model.train() -- member function of FeedForwardNN class (model.py)**

+-----------------------------------------------------------------------+
| 1)  Input parameters (inputs)                                         |
|                                                                       |
|     A.  patience                                                      |
|                                                                       |
|     B.  max\_epochs                                                   |
|                                                                       |
|     C.  lr                                                            |
|                                                                       |
|     D.  optimizer                                                     |
|                                                                       |
|     E.  validFreq                                                     |
|                                                                       |
|     F.  saveto                                                        |
|                                                                       |
|     G.  trainLoader                                                   |
|                                                                       |
|     H.  validLoader                                                   |
|                                                                       |
| 2)  Algorithm                                                         |
|                                                                       |
| [Train for max\_epochs]{.underline}**\                                |
| **- load batch\                                                       |
| - feed forward\                                                       |
| - get loss / backprop\                                                |
| - optimizer.step()                                                    |
|                                                                       |
| [Validation (at each validFreq):]{.underline}\                        |
| - validate and (epoch, error) to *stdout*\                            |
| - if it achieves the best error compared to err\_hist:\               |
| \* print to *stdout* that the latest error is the best so far\        |
| \* save model using state\_dict to the file                           |
| \<data\_prefix\>.model\_best.pkl                                      |
|                                                                       |
| +------------------------------------------------+                    |
| | **Saving model using torch.save**              |                    |
| |                                                |                    |
| | \>\> torch.save(model.state\_dict(), PATH)     |                    |
| |                                                |                    |
| | When loading,                                  |                    |
| |                                                |                    |
| | \>\> model = FeedForwardNN(\*args, \*\*kwargs) |                    |
| |                                                |                    |
| | \>\> model.load\_state\_dict(torch.load(PATH)) |                    |
| |                                                |                    |
| | \>\> model.eval()                              |                    |
| +------------------------------------------------+                    |
|                                                                       |
| \- if it does not achieve the best error and epoch is past patience:\ |
| \* increment bad\_counter\                                            |
| \* if bad\_counter exceeds patience, early stop                       |
+-----------------------------------------------------------------------+

**model.test() -- member function of FeedForwardNN class (model.py)**

+-----------------------------------------------------------------------+
| 1.  Input args\                                                       |
|     1) --test\_loader\                                                |
|     2) --data\_prefix\                                                |
|     3) --loadfrom\                                                    |
|     4)\--device                                                       |
|                                                                       |
| 2.  What to do\                                                       |
|     1) load best parameter from pkl file)                             |
|                                                                       |
| +------------------------------------------------------------------+  |
| | **Evaluation - member function of FeedForwardNN**                |  |
| |                                                                  |  |
| | 1)  Load model parameter                                         |  |
| |                                                                  |  |
| | 2)  Iterate over the test dataloader\                            |  |
| |     - load a batch\                                              |  |
| |     - put through model\                                         |  |
| |     - compare output to label and count labels and corrects;     |  |
| |     used for computing accuracy, precision, recall               |  |
| |                                                                  |  |
| | 3)  When iteration is over, compute the score and print to       |  |
| |     *stdout*                                                     |  |
| +------------------------------------------------------------------+  |
+-----------------------------------------------------------------------+
