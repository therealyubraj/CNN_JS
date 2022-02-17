class CNN {
    static kernelSize = 3;
    static poolSize = 3;
    static imgWidth = 28;
    static imgHeight = 28;

    constructor(numOfKernels, initDepth, hiddens, outputs, numOfHidden) {
        this.layers = numOfKernels.length;
        /**
         * @type {[CNNKernel[]]}
         */
        this.kernels = [];

        /**
         * @type {CNNImage[]}
         */
        this.biases = [];


        let finalDepth = numOfKernels[this.layers - 1];
        let finalWidth = CNN.imgWidth,
            finalHeight = CNN.imgHeight;
        for (let i = 0; i < numOfKernels.length; i++) {
            this.kernels.push([]);
            let kernelDepth = 0;
            if (i != 0) {
                kernelDepth = numOfKernels[i - 1];
            } else {
                kernelDepth = initDepth;
            }
            let zeroChannels = [];
            for (let j = 0; j < numOfKernels[i]; j++) {
                this.kernels[i][j] = new CNNKernel(kernelDepth, CNN.kernelSize);
                zeroChannels.push(new CNNChannel([], finalWidth, finalHeight));
                zeroChannels[j].initToRandoms();
            }

            this.biases[i] = new CNNImage(zeroChannels, finalWidth, finalHeight, numOfKernels[i]);

            finalWidth = (finalWidth + (CNN.poolSize - (finalWidth % CNN.poolSize))) / CNN.poolSize;
            finalHeight = (finalHeight + (CNN.poolSize - (finalHeight % CNN.poolSize))) / CNN.poolSize;
        }


        let NNIpSize = finalDepth * finalWidth * finalHeight;
        this.FFNN = new NN(NNIpSize, hiddens, outputs, numOfHidden);
        this.lr = 0.005;
    }


    /**
     * 
     * @param {CNNImage} img 
     * @returns {Object<string, CNNImage[]>} convolutionOutputs
     */
    applyAllConvolutions(img) {
        let convolutionOutputs = { 'convoluted': [], 'relued': [], 'pooled': [img] };

        for (let i = 0; i < this.kernels.length; i++) {
            //apply convolution
            let convolutedImage = CNNImage.applyConvolution(convolutionOutputs['pooled'][i], this.kernels[i], this.biases[i]);
            convolutionOutputs['convoluted'].push(convolutedImage);

            //Relu
            let reluedImage = CNNImage.map(convolutedImage, ReLU);
            convolutionOutputs['relued'].push(reluedImage);

            //pool
            let pooledConvolution = CNNImage.avgPool(reluedImage, CNN.poolSize);
            convolutionOutputs['pooled'].push(pooledConvolution);
        }

        //remove the original image from pooled
        convolutionOutputs['pooled'].shift();

        return convolutionOutputs;
    }

    /**
     * 
     * @param {CNNImage} img 
     * @param {boolean} drawOutputs 
     * @param {Number} dx 
     * @param {Number} dy 
     * @returns 
     */
    predict(img, drawOutputs = false, dx = 0, dy = 0) {
        /**
         * @type {CNNImage[]}
         */
        let convolutionOutputs = this.applyAllConvolutions(img).pooled;

        if (drawOutputs) {
            img.drawImage(dx, dy);
            let drawY = dy + img.height;
            rect(dx, dy, img.width, img.height);
            for (let i = 0; i < convolutionOutputs.length; i++) {
                convolutionOutputs[i].drawImage(dx, drawY);
                drawY += convolutionOutputs[i].height;
            }
        }

        let inputToNN = convolutionOutputs[convolutionOutputs.length - 1].flatten();
        let NNPrediction = this.FFNN.predict(inputToNN);
        return NNPrediction;
    }

    /**
     * 
     * @param {CNNImage} img 
     * @param {Number[]} target 
     */
    train(img, target) {
        let convolutionOutputs = this.applyAllConvolutions(img);

        let pooledOutputs = convolutionOutputs['pooled'], convolutedOutputs = convolutionOutputs['convoluted'], reluedOutputs = convolutionOutputs['relued'];
        let inputToNN = pooledOutputs[pooledOutputs.length - 1].flatten();

        //backpropagation
        let errorFromNNMatrix = this.FFNN.train(inputToNN, target, true);
        let errorFromNNArray = Matrix.toArray(Matrix.matrixMult(Matrix.transpose(this.FFNN.weights[0]), errorFromNNMatrix));

        //reshape errors from NN for CNN backprop 
        let errors = [];

        let finalErrorChannels = [];
        let finalWidth = pooledOutputs[pooledOutputs.length - 1].width,
            finalHeight = pooledOutputs[pooledOutputs.length - 1].height;
        let finalDepth = this.kernels[this.kernels.length - 1].length;
        let finalPixels = finalWidth * finalHeight;
        for (let i = 0; i < finalDepth; i++) {
            let p = errorFromNNArray.slice(finalPixels * i, finalPixels * (i + 1));
            let errorChannel = new CNNChannel(p, finalWidth, finalHeight);
            finalErrorChannels.push(errorChannel);
        }


        errors[this.layers - 1] = new CNNImage(finalErrorChannels, finalWidth, finalHeight, finalErrorChannels.length);

        for (let i = this.layers - 1; i >= 0; i--) {
            let inputImage = pooledOutputs[i - 1],
                convolutedImage = convolutedOutputs[i],
                reluedImage = reluedOutputs[i],
                pooledImage = pooledOutputs[i];

            if (i == 0) {
                inputImage = img;
            }

            let pooledError = errors[i];
            let pooledWidth = pooledImage.width, pooledHeight = pooledImage.height,
                unpooledWidth = inputImage.width, unpooledHeight = inputImage.height;

            let reluedError = CNNImage.unpool(pooledError, CNN.poolSize, unpooledWidth, unpooledHeight);
            let DReLUConvoluted = CNNImage.map(convolutedImage, DReLU);
            let convolutedError = CNNImage.mult(DReLUConvoluted, reluedError);

            //calculate error for previous layer
            if (i > 0) {
                let rotatedKernels = CNNKernel.rotate180(this.kernels[i]);
                let errorForPrev = CNNImage.applyConvolution(convolutedError, rotatedKernels);
                errorForPrev.clipImage(-255, 255);
                errors[i - 1] = errorForPrev;
            }

            //convolutedError.drawImage(0, 100 - unpooledHeight * (i - this.layers + 1));

            //calculate kernel deltas and adjust kernel values
            let calculatedKernelDeltas = CNNImage.calcKernelDeltas(convolutedError, inputImage, CNN.kernelSize);
            for (let j = 0; j < calculatedKernelDeltas.length; j++) {
                this.kernels[i][j].adjustKernels(calculatedKernelDeltas[j], this.lr);
            }
            //adjust biases values using error
            for (let j = 0; j < convolutedError.depth; j++) {
                this.biases[i].images[j].addChannel(convolutedError.images[j], this.lr);
                this.biases[i].clipImage(-255, 255);
            }
        }
    }

    serialize() {
        return JSON.stringify(this);
    }

    static deserialize(cnnObj) {
        let numKernels = [];
        for (let i = 0; i < cnnObj.kernels.length; i++) {
            numKernels.push(cnnObj.kernels[i].length);
        }

        let cnn = new CNN(numKernels, cnnObj.kernels[0][0].depth, 1, 1, 1);
        cnn.FFNN = NN.deserialize(cnnObj.FFNN);

        cnn.kernels = [];
        for (let i = 0; i < cnnObj.kernels.length; i++) {
            cnn.kernels[i] = [];
            for (let j = 0; j < cnnObj.kernels[i].length; j++) {
                cnn.kernels[i][j] = CNNKernel.deserialize(cnnObj.kernels[i][j]);
            }
        }

        cnn.biases = [];
        for (let i = 0; i < cnnObj.biases.length; i++) {
            cnn.biases[i] = CNNImage.deserialize(cnnObj.biases[i]);
        }

        return cnn;
    }
}