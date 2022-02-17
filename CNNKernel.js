class CNNKernel {
    constructor(depth, size, randomize = true) {
        this.depth = depth;
        this.size = size;
        this.kernels = [];

        for (let i = 0; i < this.depth; i++) {
            this.kernels[i] = new Matrix(this.size, this.size);
            if (randomize) {
                this.kernels[i].randomize();
                this.kernels[i].scalarMult(10);
            }
        }
    }

    /**
     * 
     * @param {[Number[]]} deltas 
     */
    adjustKernels(deltas, lr) {
        if (deltas.length != this.depth) {
            console.error('Deltas are not compatible!');
        }
        for (let i = 0; i < deltas.length; i++) {
            let deltaMatrix = Matrix.fromArray(deltas[i], this.size, this.size);
            deltaMatrix.scalarMult(lr);
            //deltaMatrix = Matrix.transpose(deltaMatrix);
            this.kernels[i].matrixSub(deltaMatrix);
            this.floorKernels();
        }
    }

    floorKernels() {
        for (let i = 0; i < this.depth; i++) {
            this.kernels[i].clip(-255, 255);
        }
    }

    scaleKernels(n) {
        for (let i = 0; i < this.kernels.length; i++) {
            this.kernels[i].scalarMult(n);
        }
    }

    /**
     * 
     * @param {CNNKernel[]} allKernels 
     */
    static rotate180(allKernels) {
        let rotatedKernels = [];
        let kernelDepth = allKernels[0].depth, kernelSize = allKernels[0].size;
        for (let i = 0; i < kernelDepth; i++) {
            let rotatedMatrices = [];

            for (let j = 0; j < allKernels.length; j++) {
                rotatedMatrices.push(allKernels[j].kernels[i].rotate180());
            }

            let rotatedKernel = new CNNKernel(rotatedMatrices.length, kernelSize, false);
            rotatedKernel.kernels = rotatedMatrices;
            rotatedKernels.push(rotatedKernel);
        }

        return rotatedKernels;
    }

    getAvgVal() {
        let avg = [];
        for (let i = 0; i < this.depth; i++) {
            avg.push(this.kernels[i].avg());
        }
        return avg;
    }

    static deserialize(kernelObj) {
        let kernel = new CNNKernel(kernelObj.depth, kernelObj.size, false);
        for (let i = 0; i < kernel.depth; i++) {
            kernel.kernels[i] = Matrix.deserialize(kernelObj.kernels[i]);
        }
        return kernel;
    }
}