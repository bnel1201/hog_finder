## need to use different method of loading and saving models because

- load_learner is not cross-compatible between windows and linux


```python
AttributeError: 'dict' object has no attribute 'dls'
Traceback:
File "/usr/local/lib/python3.8/dist-packages/streamlit/script_runner.py", line 333, in _run_script
    exec(code, module.__dict__)
File "/code/streamlit_app.py", line 50, in <module>
    main()
File "/code/streamlit_app.py", line 29, in main
    show_image(imlist[selected_frame_index], show_mask, predict)
File "/code/streamlit_app.py", line 44, in show_image
    hf = hedgiefinder.HedgieFinder(fname, cleanup=False).predict([fname])
File "/code/hedgiefinder/inference.py", line 33, in __init__
    self.model = load_learner(model_dir/model_name)
File "/usr/local/lib/python3.8/dist-packages/fastai/learner.py", line 376, in load_learner
    if cpu: res.dls.cpu()
```

Potential solution:
load_learner expects an output from learn.export. Not learn.save

<https://forums.fast.ai/t/load-model-from-pth-file/37440/4>