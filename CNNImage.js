class CNNImage {
    /** 
    * @param {CNNChannel[]} images
    */
    constructor(images, width, height, depth) {
        if (images.length != depth) {
            console.error("CNNImage Error: Pixels array depth doesnot match depth value");
            return;
        }
        /**
         * @type {CNNChannel[]}
         */
        this.images = images;
        this.width = width;
        this.height = height;
        this.depth = depth;
    }

    drawImage(x, y) {
        for (let i = 0; i < this.images.length; i++) {
            this.images[i].drawChannel(x + i * this.width, y);
        }
    }

    flatten() {
        let flattened = [];
        for (let i = 0; i < this.images.length; i++) {
            for (let j = 0; j < this.images[i].pixels.length; j++) {
                flattened.push(this.images[i].pixels[j] / 255);
            }
        }
        return flattened;
    }

    map(f) {
        for (let i = 0; i < this.images.length; i++) {
            this.images[i].mapPixels(f);
        }
    }

    calcPartialError(errors) {
        let partialErrorChannels = [];
        for (let i = 0; i < errors.length; i++) {
            partialErrorChannels.push(this.images[i].calcPartialError(errors[i]));
        }
        let partialErrorImage = new CNNImage(partialErrorChannels, this.width, this.height, errors.length);
        return partialErrorImage;
    }

    clipImage(min, max) {
        for (let i = 0; i < this.depth; i++) {
            this.images[i].clipValues(min, max);
        }
    }

    /**
     * 
     * @param {CNNImage} img 
     * @param {CNNKernel} allKernels 
     * @param {CNNImage} biases 
     * @returns 
     */
    static applyConvolution(img, allKernels, biases) {
        if (allKernels[0].depth != img.depth) {
            console.error("Image and Kernels are not compatible!");
        }

        let outputs = [];
        for (let i = 0; i < allKernels.length; i++) {
            let kernel = allKernels[i];
            let bias;

            if (biases) {
                bias = biases.images[i];
            }

            let kernelOutput = new CNNChannel([], img.width, img.height);
            kernelOutput.initToZeros();

            for (let j = 0; j < img.depth; j++) {
                let feature = img.images[j];
                let filterMatrix = kernel.kernels[j];

                let convolvedFeature = CNNChannel.convolveImage(feature, filterMatrix);
                kernelOutput.addChannel(convolvedFeature);
                // convolvedFeature.drawChannel(j * 28, 40);
            }
            if (bias) {
                kernelOutput.addChannel(bias);
            }
            outputs.push(kernelOutput);
        }

        let outputImage = new CNNImage(outputs, img.width, img.height, allKernels.length);
        return outputImage;
    }

    /**
     * @param {CNNImage} img Image to pool
     * @param {avgPoolSize} avgPoolSize pool size
     */
    static avgPool(img, avgPoolSize) {
        let newWidth = (img.width + (avgPoolSize - (img.width % avgPoolSize))) / avgPoolSize;
        let newHeight = (img.height + (avgPoolSize - (img.height % avgPoolSize))) / avgPoolSize;

        let pooledChannels = [];

        for (let i = 0; i < img.images.length; i++) {
            let pooled = new CNNChannel([], newWidth, newHeight);
            for (let y = 0; y < newHeight * avgPoolSize; y += avgPoolSize) {
                for (let x = 0; x < newWidth * avgPoolSize; x += avgPoolSize) {
                    let avgVal = 0;
                    for (let ny = y; ny < y + avgPoolSize; ny++) {
                        for (let nx = x; nx < x + avgPoolSize; nx++) {
                            avgVal += img.images[i].getFromPixel(nx, ny);
                        }
                    }

                    avgVal /= avgPoolSize ** 2;

                    pooled.pixels.push(avgVal);
                }
            }
            pooledChannels.push(pooled);
        }
        let pooledImage = new CNNImage(pooledChannels, newWidth, newHeight, img.depth);
        return pooledImage;
    }

    static unpool(img, poolSize, unpooledWidth, unpooledHeight) {
        let unpooledChannels = [];

        for (let i = 0; i < img.images.length; i++) {
            unpooledChannels.push(CNNChannel.unpool(img.images[i], poolSize, unpooledWidth, unpooledHeight));
        }

        let unpooledImage = new CNNImage(unpooledChannels, unpooledWidth, unpooledHeight, img.images.length);
        return unpooledImage;
    }

    static map(img, f) {
        let mappedChannels = [];
        for (let i = 0; i < img.depth; i++) {
            mappedChannels.push(CNNChannel.map(img.images[i], f));
        }

        let mappedImage = new CNNImage(mappedChannels, img.width, img.height, img.depth);
        return mappedImage;
    }

    static mult(img1, img2) {
        if (img1.depth != img2.depth) {
            console.error("Incompatible images to multiply!");
            return;
        }

        let multipliedChannels = [];

        for (let i = 0; i < img1.depth; i++) {
            multipliedChannels.push(CNNChannel.mult(img1.images[i], img2.images[i]));
        }

        let multipliedImg = new CNNImage(multipliedChannels, img1.width, img2.height, img1.depth);
        return multipliedImg;
    }

    /**
     * 
     * @param {CNNImage} errorImg 
     * @param {CNNImage} inputImg 
     */
    static calcKernelDeltas(errorImg, inputImg, kernelSize) {
        let deltas = [];
        for (let i = 0; i < errorImg.depth; i++) {
            let error = errorImg.images[i];

            // let partialErrorChannels = [];
            // for (let i = 0; i < inputImg.depth; i++) {
            //     partialErrorChannels[i] = new CNNChannel([], inputImg.width, inputImg.height);
            // }

            // for (let i = 0; i < inputImg.images[0].pixels.length; i++) {
            //     let sum = 0;
            //     for (let j = 0; j < inputImg.depth; j++) {
            //         sum += inputImg.images[j].pixels[i];
            //     }
            //     if (sum == 0) {
            //         sum = 1;
            //     }

            //     for (let j = 0; j < inputImg.depth; j++) {
            //         partialErrorChannels[j].pixels.push(inputImg.images[j].pixels[i] * error.pixels[i] / sum);
            //     }
            // }

            // let partialErrorImage = new CNNImage(partialErrorChannels, inputImg.width, inputImg.height, inputImg.depth);
            //partialErrorImage.drawImage(30 * i, 70);
            deltas.push([]);
            for (let j = 0; j < inputImg.depth; j++) {
                let ip = inputImg.images[j];
                // let e = partialErrorImage.images[j];
                let delta = CNNChannel.calcKernelDeltas(error, ip, kernelSize);
                deltas[i].push(delta);
            }
        }
        return deltas;
    }

    static deserialize(imageObj) {
        let channels = [];
        for (let i = 0; i < imageObj.images.length; i++) {
            channels.push(CNNChannel.deserialize(imageObj.images[i]));
        }

        return new CNNImage(channels, imageObj.width, imageObj.height, imageObj.depth);
    }
}