function view_images(images)
        for i=1:length(images)
            figure;
            imshow(images{i});
        end