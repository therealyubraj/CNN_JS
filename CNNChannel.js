function getXYFromIndex(i, width) {
    let x = i % (width);
    let y = (i - x) / width;
    return [x, y];
}

class CNNChannel {
    /** @param {Number[]} pixels */
    constructor(pixels, width, height) {
        this.pixels = pixels;
        this.width = width;
        this.height = height;
    }

    mapPixels(f) {
        this.pixels = this.pixels.map(p => f(p));
    }

    initToZeros() {
        for (let i = 0; i < this.width * this.height; i++) {
            this.pixels[i] = 0;
        }
    }

    initToRandoms() {
        for (let i = 0; i < this.width * this.height; i++) {
            this.pixels[i] = Math.random() * 0;
        }
    }

    addChannel(channelToAdd, lr = 1) {
        if (this.width != channelToAdd.width || this.height != channelToAdd.height || this.pixels.length != channelToAdd.pixels.length) {
            console.error("Cannot add images! They are imcompatible");
            console.error(this, channelToAdd);
            return;
        }

        for (let i = 0; i < this.pixels.length; i++) {
            let inc = channelToAdd.pixels[i] * lr;
            this.pixels[i] += inc;
        }
    }

    clipValues(min, max) {
        for (let i = 0; i < this.pixels.length; i++) {
            if (this.pixels[i] < min) this.pixels[i] = min;
            if (this.pixels[i] > max) this.pixels[i] = max;
        }
    }

    drawChannel(x, y) {
        let img = createImage(this.width, this.height);
        img.loadPixels();
        for (let i = 0; i < this.pixels.length; i++) {
            img.pixels[i * 4 + 0] = this.pixels[i];
            img.pixels[i * 4 + 1] = this.pixels[i];
            img.pixels[i * 4 + 2] = this.pixels[i];
            img.pixels[i * 4 + 3] = 255;
        }
        img.updatePixels();
        image(img, x, y);
        noFill();
        strokeWeight(1);
        stroke(255);
        rect(x, y, this.width, this.height);
    }

    getFromPixel(x, y) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) return 0;
        return this.pixels[x + y * this.width];
    }

    setPixel(x, y, val) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) return;
        this.pixels[x + y * this.width] = val;
    }

    scalarAdd(n) {
        this.pixels = this.pixels.map(p => p + n);
    }

    calcPartialError(e) {
        let sum = this.pixels.reduce((a, b) => a + b);
        let errorChannel = new CNNChannel([], this.width, this.height);
        for (let i = 0; i < this.pixels.length; i++) {
            let partialError = (this.pixels[i] * e) / sum;
            if (sum == 0) {
                partialError = 0.01;
            }
            errorChannel.pixels.push(partialError);
        }
        return errorChannel;
    }

    static convolveImage(channel, kernel) {
        let newChannel = new CNNChannel([], channel.width, channel.height);
        for (let y = 0; y < channel.height; y++) {
            for (let x = 0; x < channel.width; x++) {
                let padding = Math.floor(kernel.width / 2);
                let selectedPixels = [];
                for (let ny = y - padding; ny <= y + padding; ny++) {
                    for (let nx = x - padding; nx <= x + padding; nx++) {
                        let val = channel.getFromPixel(nx, ny);
                        selectedPixels.push(val);
                    }
                }
                let filteredMatrix = Matrix.hadamardMult(kernel, Matrix.fromArray(selectedPixels, kernel.width, kernel.height));

                let newVal = Matrix.toArray(filteredMatrix).reduce((a, b) => a + b);

                if (isNaN(newVal)) {
                    console.error("Convolution returned NaN values!");
                }

                newChannel.pixels.push(newVal);
            }
        }
        return newChannel;
    }

    static unpool(img, poolSize, unpooledWidth, unpooledHeight) {
        let unpooledChannel = new CNNChannel([], unpooledWidth, unpooledHeight);
        for (let x = 0; x < img.width; x++) {
            for (let y = 0; y < img.height; y++) {
                let pooledVal = img.getFromPixel(x, y);

                for (let nx = x * poolSize; nx < (x * poolSize) + poolSize; nx++) {
                    for (let ny = y * poolSize; ny < (y * poolSize) + poolSize; ny++) {
                        unpooledChannel.setPixel(nx, ny, pooledVal);
                    }
                }
            }
        }

        return unpooledChannel;
    }

    static map(img, f) {
        let mapped = new CNNChannel([], img.width, img.height);

        for (let i = 0; i < img.pixels.length; i++) {
            mapped.pixels[i] = f(img.pixels[i]);
        }

        return mapped;
    }

    static mult(img1, img2) {
        let mapped = new CNNChannel([], img1.width, img1.height);

        for (let i = 0; i < img1.pixels.length; i++) {
            mapped.pixels[i] = img1.pixels[i] * img2.pixels[i];
        }

        return mapped;
    }

    /**
     * 
     * @param {CNNChannel} errorImg 
     * @param {CNNChannel} inputImg 
     */
    static calcKernelDeltas(errorImg, inputImg, kernelSize) {
        let deltas = [];

        for (let n = 0; n < kernelSize; n++) {
            for (let m = 0; m < kernelSize; m++) {
                let hadSum = 0;
                for (let i = 0; i < errorImg.pixels.length; i++) {
                    let xyFromI = getXYFromIndex(i, errorImg.width);
                    let iX = xyFromI[0],
                        iY = xyFromI[1];

                    let aX = iX + (m - 1),
                        aY = iY + (n - 1);
                    let eVal = errorImg.pixels[i],
                        aVal = inputImg.getFromPixel(aX, aY);
                    let inc = eVal * aVal;
                    hadSum += inc;
                }
                // hadSum *= 1 / (errorImg.pixels.length);
                deltas.push(hadSum);
            }
        }
        return deltas;
    }

    static deserialize(channelObj) {
        let channel = new CNNChannel(channelObj.pixels, channelObj.width, channelObj.height);
        return channel;
    }
}