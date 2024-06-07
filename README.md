# Experimental plan and literature review

## Research question

How many representative objects need to be scanned with (industrial setting) x-ray scanning in order to allow good deep learning based segmentation of foreign objects embedded in products.

<!-- # TODO: How does the accuracy of deep-learning based foreign object detection using method ... depend on the number of training examples? -->

## Experimental plan

In short: the experimental plan is to follow the pipeline shown in the figure below for differing amounts of objects, in order to create differing amounts of training data. This training data will then be used to train a deep network for image segmentation. Performance of this network will be evaluated on a test set.

![The complete workflow of data acquisition (a, b) and the generation of training data (c, h) for deep learning driven foreign object detection, through 3D reconstruction from the CT scan (d, e), segmentation (f), and virtual projections (g). The reconstruction reveals the hidden foreign objects inside the main object. Note that the projection data (d) is usually just the set of fast-acquisition radiographs (d).](readme_media/workflow.png)

In more detail:

1. Create a specific amount of phantoms (in the range 1 to 100)
2. Do virtual fast-acquisition scanning by using the Astra toolbox
    - The used scanning settings will be directly taken from the paper
    - They are meant to represent fast-acquisition x-ray scanning in an industrial setting
3. Use the generated projection data to create ground truth 3D locations of the foreign objects
    - Done by reconstructing the projection data
    - Then segmenting the foreign object
    - And doing fast-acquisition scanning using the Astra toolbox of just the foreign object
4. Use the created training set (consisting of the full object radiographs and the true foreign object location radiographs) to train a deep learning segmentation network
    - This will probably be an altered version of U-Net as described in the paper
5. Test the network accuracy for a predefined test set

The plan is to try as many amounts of training phantoms (the phantoms that will be used to generate training data) as possible within the timeframe for this assignment. This will be heavily dependent on the computational cost of the experiment described above.

<!-- # TODO: Which base and foreign object phantoms will you use and what will be the image size -->

## Literature review

> **Note**: the headings are links to the mentioned papers

### [Implementation paper](https://www.sciencedirect.com/science/article/pii/S0957417422010429)
<!-- 
```bibtex
@article{ZEEGERS2022117768,
title = {A tomographic workflow to enable deep learning for X-ray based foreign object detection},
journal = {Expert Systems with Applications},
volume = {206},
pages = {117768},
year = {2022},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2022.117768},
url = {https://www.sciencedirect.com/science/article/pii/S0957417422010429},
author = {Mathé T. Zeegers and Tristan {van Leeuwen} and Daniël M. Pelt and Sophia Bethany Coban and Robert {van Liere} and Kees Joost Batenburg},
keywords = {X-ray imaging, Foreign object detection, Segmentation, Computed tomography, Machine learning, Deep learning},
abstract = {Detection of unwanted (‘foreign’) objects within products is a common procedure in many branches of industry for maintaining production quality. X-ray imaging is a fast, non-invasive and widely applicable method for foreign object detection. Deep learning has recently emerged as a powerful approach for recognizing patterns in radiographs (i.e., X-ray images), enabling automated X-ray based foreign object detection. However, these methods require a large number of training examples and manual annotation of these examples is a subjective and laborious task. In this work, we propose a Computed Tomography (CT) based method for producing training data for supervised learning of foreign object detection, with minimal labor requirements. In our approach, a few representative objects are CT scanned and reconstructed in 3D. The radiographs that are acquired as part of the CT-scan data serve as input for the machine learning method. High-quality ground truth locations of the foreign objects are obtained through accurate 3D reconstructions and segmentations. Using these segmented volumes, corresponding 2D segmentations are obtained by creating virtual projections. We outline the benefits of objectively and reproducibly generating training data in this way. In addition, we show how the accuracy depends on the number of objects used for the CT reconstructions. The results show that in this workflow generally only a relatively small number of representative objects (i.e., fewer than 10) are needed to achieve adequate detection performance in an industrial setting.}
}
``` -->

The implementation paper upon which the final paper will mostly be based. It describes important settings for phantom generation, radiograph generation, foreign object segmentation and deep learning based segmentation.

### [ASTRA toolbox](https://opg.optica.org/oe/abstract.cfm?URI=oe-24-22-25129)
<!-- 
```bibtex
@article{vanAarle:16,
author = {Wim van Aarle and Willem Jan Palenstijn and Jeroen Cant and Eline Janssens and Folkert Bleichrodt and Andrei Dabravolski and Jan De Beenhouwer and K. Joost Batenburg and Jan Sijbers},
journal = {Opt. Express},
keywords = {Tomographic image processing; Computational imaging ; Tomographic imaging ; Computed tomography; Detector arrays; Detectors; Image reconstruction; Imaging techniques; Phase retrieval},
number = {22},
pages = {25129--25147},
publisher = {Optica Publishing Group},
title = {Fast and flexible X-ray tomography using the ASTRA toolbox},
volume = {24},
month = {Oct},
year = {2016},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-24-22-25129},
doi = {10.1364/OE.24.025129},
abstract = {Object reconstruction from a series of projection images, such as in computed tomography (CT), is a popular tool in many different application fields. Existing commercial software typically provides sufficiently accurate and convenient-to-use reconstruction tools to the end-user. However, in applications where a non-standard acquisition protocol is used, or where advanced reconstruction methods are required, the standard software tools often are incapable of computing accurate reconstruction images. This article introduces the ASTRA Toolbox. Aimed at researchers across multiple tomographic application fields, the ASTRA Toolbox provides a highly efficient and highly flexible open source set of tools for tomographic projection and reconstruction. The main features of the ASTRA Toolbox are discussed and several use cases are presented.},
}
``` -->

Paper introducing the ASTRA toolbox. We will use the ASTRA toolbox's rotating cone-beam geometry to create radiographs from phantoms. We will use the ASTRA toolbox's SIRT implementation with 100 iterations for the reconstruction.

### [Foreign object segmentation](https://www.sciencedirect.com/science/article/pii/S1076633219303538)
<!-- 
```bibtex
@article{lenchik2019automated,
  title={Automated segmentation of tissues using CT and MRI: a systematic review},
  author={Lenchik, Leon and Heacock, Laura and Weaver, Ashley A and Boutin, Robert D and Cook, Tessa S and Itri, Jason and Filippi, Christopher G and Gullapalli, Rao P and Lee, James and Zagurovskaya, Marianna and others},
  journal={Academic radiology},
  volume={26},
  number={12},
  pages={1695--1706},
  year={2019},
  publisher={Elsevier},
  url={https://www.sciencedirect.com/science/article/pii/S1076633219303538}
}
``` -->

In order to get ground truth for the foreign object locations, we need to segment the foreign objects from the reconstructed objects. This paper describes automated segmentation for computed tomography. The paper mainly focusses on medical imaging implementations where segmentation is mostly based on tissue, but since our phantoms simulate foreign objects by using a different density (ex: flesh object with a bone object embedded in it) we can use the approaches described in this paper.

### [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
<!-- 
```bibtex
@InProceedings{10.1007/978-3-319-24574-4_28,
author="Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas",
editor="Navab, Nassir and Hornegger, Joachim and Wells, William M.
and Frangi, Alejandro F.",
title="U-Net: Convolutional Networks for Biomedical Image Segmentation",
booktitle="Medical Image Computing and Computer-Assisted Intervention -- MICCAI 2015",
year="2015",
publisher="Springer International Publishing",
address="Cham",
pages="234--241",
abstract="There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net.",
isbn="978-3-319-24574-4"
}
``` -->

This paper describes U-Net: a deep learning based segmentation method that aims to be as accurate as possible while using as little training data as possible. We will be using a modified implementation of U-Net as our deep learning based segmentation method to test segmentation performance based on the amount of training data.

<!-- https://tomroelandts.com/ -->

<!-- Structure of the paper:
