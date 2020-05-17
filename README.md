# colour-term-grounding

Code for the ACL 2020 paper

1) extract objects and annotations
  - preprocessing/data_extraction.py

2) train/test/dev splits
  - preprocessing/data_split.py

3) extract histograms
  - preprocessing/histogram_colour/extract_histograms.py

4) filter + resample data
  - preprocessing/data_resampling.py

5) train models
  - models/bottomup/bottomup-train.py
  - models/topdown/topdown-train.py
  - models/earlyfusion/earlyfusion-train.py

6) get model predictions
  - models/bottomup/bottomup-predict.py
  - models/topdown/topdown-predict.py
  - models/earlyfusion/earlyfusion-predict.py

7) evaluation
  - evaluation/evaluation.ipynb
  - evaluation/evaluation_unseen.ipynb
