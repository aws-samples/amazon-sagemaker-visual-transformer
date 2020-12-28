## Image Classification using Visual Transformers
 - [Overview](#overview)
  - [Amazon SageMaker](#-amazon-sagemaker)
  - [How to run the code in Amazon SageMaker Studio?](#-how-to-run-the-code-in-amazon-sagemaker-studio)
  - [References](#-references)

In standard image classification algorithms like ResNet, InceptionNet etc., images are represented as pixel arrays on which a series of convolution operations are performed. Although, great accuracy has been achieved with these algorithms, the convolution operation is computationally expensive. Therefore, in this notebook we will look at an alternative way to perform `Image Classification` using the ideas mentioned in the `Visual Transformers: Token-based Image Representation and Processing for Computer Vision` [research paper](https://arxiv.org/pdf/2006.03677.pdf). 

<img src=./img/vt.png width="600" height="300">

Diagram of a Visual Transformer (VT). 
<br>
For a given image, we first apply convolutional layers to extract low-level
features. The output feature map is then fed to VT: First, apply a tokenizer, grouping pixels into a small number of visual
tokens. Second, apply transformers to model relationships between tokens.
Third, visual tokens are directly used for image classification or projected back to the feature map for semantic segmentation.

**Note** 
- Dataset used is **Intel Image Classification** from [Kaggle](https://www.kaggle.com/puneet6060/intel-image-classification).
- The notebook is only an example and not to be used for production deployments.
- Use `Python3 (PyTorch 1.6 Python 3.6 CPU Optimized)` kernel and `ml.m5.large (2 vCPU + 8 GiB)` for the notebook, if you are using Amazon SageMaker Studio.
- Notebook has ideas and some of the pseudo code from `Visual Transformers: Token-based Image Representation and Processing for Computer Vision` [research paper](https://arxiv.org/pdf/2006.03677.pdf) but does not reproduces the results mentioned in the paper. 

## Amazon SageMaker
----
Amazon SageMaker is the most comprehensive and full managed machine learning service. With SageMaker, data scientists and developers can quickly and easily build and train machine learning models, and then directly deploy them into a production-ready hosted environment. It provides an integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis, so you don't have to manage servers. It also provides common machine learning algorithms that are optimized to run efficiently against extremely large data in a distributed environment. With native support for bring-your-own-algorithms and frameworks, SageMaker offers flexible distributed training options that adjust to your specific workflows. Deploy a model into a secure and scalable environment by launching it with a few clicks from SageMaker Studio or the SageMaker console.  We use Amazon SageMaker Studio for running the code, for more details see the [AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html).

## How to run the code in Amazon SageMaker Studio? 
----
If you haven't used Amazon SageMaker Studio before, please follow the steps mentioned in [`Onboard to Amazon SageMaker Studio`](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html).

### To log in from the SageMaker console

- Onboard to Amazon SageMaker Studio. If you've already onboarded, skip to the next step.
- Open the SageMaker console.
- Choose Amazon SageMaker Studio.
- The Amazon SageMaker Studio Control Panel opens.
- In the Amazon SageMaker Studio Control Panel, you'll see a list of user names.
- Next to your user name, choose Open Studio.

### Open a Studio notebook
SageMaker Studio can only open notebooks listed in the Studio file browser. In this example we will `Clone a Git Repository in SageMaker Studio`.

#### To clone the repo

- In the left sidebar, choose the File Browser icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/File_browser_squid.png'> ).
- Choose the root folder or the folder you want to clone the repo into.
- In the left sidebar, choose the Git icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/Git_squid.png'>  ).
- Choose Clone a Repository.
- Enter the URI for the repo https://github.com/aws-samples/amazon-sagemaker-visual-transformer.
- Choose CLONE.
- If the repo requires credentials, you are prompted to enter your username and password.
- Wait for the download to finish. After the repo has been cloned, the File Browser opens to display the cloned repo.
- Double click the repo to open it.
- Choose the Git icon to view the Git user interface which now tracks the examples repo.
- To track a different repo, open the repo in the file browser and then choose the Git icon.

### To open a notebook

- In the left sidebar, choose the File Browser icon ( <img src='https://docs.aws.amazon.com/sagemaker/latest/dg/images/icons/File_browser_squid.png'> ) to display the file browser.
- Browse to a notebook file and double-click it to open the notebook in a new tab.

## References
----
 - Visual Transformers: Token-based Image Representation and Processing for
Computer Vision (https://arxiv.org/pdf/2006.03677.pdf).
- Kaggle notebook (https://www.kaggle.com/asollie/intel-image-multiclass-pytorch-94-test-acc).
- Registry of Open Data from AWS (https://registry.opendata.aws/multimedia-commons/).


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

