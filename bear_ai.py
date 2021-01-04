from fastai.vision.all import *
path = Path("bears")

# fns = get_image_files(path)
# print(fns)
# failed = verify_images(fns)
# print(failed)

bears = DataBlock(
	blocks=(ImageBlock, CategoryBlock),
	get_items=get_image_files,
	splitter=RandomSplitter(valid_pct=0.3, seed=42),
	get_y=parent_label,
	item_tfms=Resize(300),
	batch_tfms=aug_transforms(mult=2))

dls = bears.dataloaders(path, num_workers=0)
# dls.valid.show_batch(max_n=4, nrows=1)

# Train the model
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(4)

# Interprate the results
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

interp.plot_top_losses(5, nrows=1)

# Does not work
# cleaner = ImageClassifierCleaner(learn)
# cleaner


# # Export the model
# learn.export()
# path=Path()
# path.ls(file_exts='.pkl')


# # Run the exported file as a program, on another server for a example
# # Server inference
# learn_inf = load_learner(path/'export.pkl')
# learn_inf.dls.vocab
# learn_inf.predict('images/grizzly1.jpg')