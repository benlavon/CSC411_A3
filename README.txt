
File descriptions

You should use only "labeled_images.mat" and "unlabeled_images.mat" to learn your model. Then you should apply your learned model on public_test_images.mat in order to generate a submission.

    labeled_images.mat - the labeled images for learning your model. 
        tr_identity - an anonymous identifier unique to a given individual. This is not the image Id.
        tr_labels - the labels for each image.
        tr_images - the images given by pixel matrices (32 pixels by 32 pixels by 2925 images)
    public_test_images.mat - the public test set. Predictions on this set will be scored on the leaderboard.
        public_test_images - the images given by pixel marries (32 pixels by 32 pixels by 418 images)
    sample_submission.csv - a sample submission file in the correct format.
    kNN_baseline.zip - baseline classifier.
    unlabeled_images.mat - additional unlabeled images that you may want to use.
        unlabeled_images - the images given by pixel marries (32 pixels by 32 pixels by 98058 images)


