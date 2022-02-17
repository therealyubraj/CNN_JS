let mnistTrainImages, mnistTrainLabels, mnistTestImages, mnistTestLabels;
let initChannels = 3;
/**
 * @type {CNN}
 */
let brain;
let nn;

let trainIter = 1, testIter = 1;
let trainer = [], tester = [];

function preload() {
    mnistTrainImages = loadBytes('train-images.idx3-ubyte');
    mnistTrainLabels = loadBytes('train-labels.idx1-ubyte');
    mnistTestImages = loadBytes('t10k-images.idx3-ubyte');
    mnistTestLabels = loadBytes('t10k-labels.idx1-ubyte');
}

function setup() {
    createCanvas(300, 300);

    background(0);
    nn = new NN(2, 10, 1, 2);
    let allTrainImages = loadMNISTImages(mnistTrainImages.bytes, 16, 60000);
    let allTrainLabels = loadMNISTLabels(mnistTrainLabels.bytes, 8);
    let allTestImages = loadMNISTImages(mnistTestImages.bytes, 16, 10000);
    let allTestLabels = loadMNISTLabels(mnistTestLabels.bytes, 8);

    let testButton = select('#testButton');
    let trainButton = select('#trainButton');
    let showButton = select('#showButton');
    let clearButton = select('#clearButton');
    let saveButton = select('#saveButton');
    let loadButton = select('#loadButton');

    testButton.mousePressed(function () {
        testBrain(testIter);
    });
    trainButton.mousePressed(function () {
        trainBrain(trainIter);
    });
    showButton.mousePressed(function () {
        showSampleConvolution();
    });
    clearButton.mousePressed(function () {
        background(0);
    });
    saveButton.mousePressed(function () {
        saveJSON(brain, 'brain.json');
    });
    loadButton.mousePressed(function () {
        loadJSON('brain.json', function (data) {
            brain = CNN.deserialize(data);
        });
    });

    allTrainImages.forEach((img, i) => {
        let t = new Trainer(img, allTrainLabels[i]);
        trainer.push(t);
    });
    allTestImages.forEach((img, i) => {
        let t = new Trainer(img, allTestLabels[i]);
        tester.push(t);
    });

    // let i = trainer[0].img;
    // let k = new CNNKernel(3, 3, false);
    // k.kernels[0].matrix = [0, 0, 0, 0, 1, 0, 0, 0, 0];
    // k.kernels[1].matrix = [0, 0, 0, 0, 1, 0, 0, 0, 0];
    // k.kernels[2].matrix = [0, 0, 0, 0, 1, 0, 0, 0, 0];
    // let op = CNNImage.applyConvolution(i, [k]);
    // console.log(op);
    // op.drawImage(0, 0);

    //console.log(allTrainImages);
    brain = new CNN([10], initChannels, 30, 10, 2);
}

function draw() {

}


function loadMNISTImages(pixelsArray, offest, totalImages) {
    let images = [];
    for (let imageInd = 0; imageInd < totalImages; imageInd++) {
        let imageOffset = 28 * 28 * imageInd;
        let mI = offest + imageOffset;
        let mINext = mI + 784;
        let imgPixels = [];
        for (let i = mI; i < mINext; i++) {
            imgPixels.push(pixelsArray[i]);
        }
        // imgPixels = imgPixels.map(p => {
        //     if (p != 0) p = 255;
        //     return (p / 255);
        // });
        imgPixels = greyToMany(imgPixels, initChannels);

        let imgChannels = [];
        imgPixels.forEach((pixels) => {
            let channel = new CNNChannel(pixels, 28, 28);
            imgChannels.push(channel);
        });
        let img = new CNNImage(imgChannels, 28, 28, imgChannels.length);
        images.push(img);
    }
    return images;
}

function loadMNISTLabels(labelsArray, offset) {
    let labels = [];
    for (let i = offset; i < labelsArray.length; i++) {
        let l = [];
        for (let j = 0; j < 10; j++) {
            l[j] = 0;
        }
        l[labelsArray[i]] = 1;
        labels.push(l);
    }
    return labels;
}

function showSampleConvolution() {
    brain.predict(trainer[0].img, true, 0, 0);
}

function trainBrain(n) {
    shuffle(trainer, true);
    for (let k = 1; k <= n; k++) {
        console.log('Training!');
        for (let i = 0; i < 10000; i++) {
            let rI = trainer[i];
            brain.train(rI.img, rI.label);
            console.log('0');
        }
        console.log(k + " Epoch has finished!");
        testBrain();
        // showSampleConvolution();
    }
    //brain.getKernelsAvg();
}

function testBrain() {
    let correct = 0;
    let totalTests = 5000;
    console.log('Testing!');
    for (let i = 0; i < totalTests; i++) {
        let rI = random(tester);
        let prediction = brain.predict(rI.img);
        let highestInd = prediction.indexOf(Math.max(...prediction));
        let actualVal = rI.label.indexOf(1);
        if (actualVal == highestInd) {
            correct++;
        }
        console.log('0');
    }
    console.log((correct / totalTests) * 100 + '%');
    showSampleConvolution();
}

function greyToMany(greyPixels, n) {
    let nPixels = [];
    for (let i = 0; i < n; i++) {
        nPixels.push(greyPixels.slice(0, greyPixels.length));
    }
    return nPixels;
}

function testNN() {
    for (let i = 0; i < 10000; i++) {
        let x = 0, y = 0;
        if (random() < 0.5) x = 1;
        if (random() < 0.5) y = 1;
        let op;
        if (x == y) op = 0;
        if (x != y) op = 1;

        nn.train([x, y], [op]);
    }
    console.log(nn.predict([0, 0])[0]);
    console.log(nn.predict([1, 1])[0]);
    console.log(nn.predict([1, 0])[0]);
    console.log(nn.predict([0, 1])[0]);
}
