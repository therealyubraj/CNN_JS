let mnistTrainImages, mnistTrainLabels, mnistTestImages, mnistTestLabels;
let initChannels = 3;
/**
 * @type {CNN}
 */
let brain;
let nn;

let trainIter = 1,
  testIter = 1;
let trainer = [],
  tester = [];

function preload() {
  console.log("Preloaded");
  mnistTrainImages = loadBytes("train-images.idx3-ubyte");
  mnistTrainLabels = loadBytes("train-labels.idx1-ubyte");
  mnistTestImages = loadBytes("t10k-images.idx3-ubyte");
  mnistTestLabels = loadBytes("t10k-labels.idx1-ubyte");
}

function setup() {
  const canvas = createCanvas(280, 280);
  canvas.parent("canvas-holder");

  background(0);
  nn = new NN(2, 10, 1, 2);
  let allTrainImages = loadMNISTImages(mnistTrainImages.bytes, 16, 60000);
  let allTrainLabels = loadMNISTLabels(mnistTrainLabels.bytes, 8);
  let allTestImages = loadMNISTImages(mnistTestImages.bytes, 16, 10000);
  let allTestLabels = loadMNISTLabels(mnistTestLabels.bytes, 8);

  //let testButton = select("#testButton");
  //let trainButton = select("#trainButton");
  //let showButton = select("#showButton");
  //let saveButton = select("#saveButton");
  let clearButton = select("#clearButton");
  let guessButton = select("#guessButton");

  drawingContext.willReadFrequently = true;

  //testButton.mousePressed(function () {
  //testBrain(testIter);
  //});
  //trainButton.mousePressed(function () {
  //trainBrain(trainIter);
  //});
  //showButton.mousePressed(function () {
  //showSampleConvolution();
  //});
  //saveButton.mousePressed(function () {
  //saveJSON(brain, "brain.json");
  //});
  loadJSON("brain.json", function (data) {
    brain = CNN.deserialize(data);
  });
  clearButton.mousePressed(function () {
    background(0);
  });
  guessButton.mousePressed(function () {
    guessFromScreen();
  });

  allTrainImages.forEach((img, i) => {
    let t = new Trainer(img, allTrainLabels[i]);
    trainer.push(t);
  });
  allTestImages.forEach((img, i) => {
    let t = new Trainer(img, allTestLabels[i]);
    tester.push(t);
  });

  console.log(allTrainImages);
  brain = new CNN([10], initChannels, 30, 10, 2);
}

function draw() {
  if (mouseIsPressed) {
    stroke(255);
    strokeWeight(15);
    line(pmouseX, pmouseY, mouseX, mouseY);
  }
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
    //imgPixels = imgPixels.map((p) => {
    //if (p != 0) p = 255;
    //return p / 255;
    //});
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
  brain.predict(random(trainer).img, true, 0, 0);
}

function trainBrain(n) {
  shuffle(trainer, true);
  for (let k = 1; k <= n; k++) {
    console.log("Training!");
    for (let i = 0; i < 10000; i++) {
      let rI = trainer[i];
      brain.train(rI.img, rI.label);
      console.log("0");
    }
    console.log(k + " Epoch has finished!");
    testBrain();
    showSampleConvolution();
  }
  //brain.getKernelsAvg();
}

function testBrain() {
  let correct = 0;
  let totalTests = 5000;
  console.log("Testing!");
  for (let i = 0; i < totalTests; i++) {
    let rI = random(tester);
    let prediction = brain.predict(rI.img);
    let highestInd = prediction.indexOf(Math.max(...prediction));
    let actualVal = rI.label.indexOf(1);
    if (actualVal == highestInd) {
      correct++;
    }
    console.log("0");
  }
  console.log((correct / totalTests) * 100 + "%");
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
    let x = 0,
      y = 0;
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

function guessFromScreen() {
  loadPixels();

  let minX = width,
    minY = height;
  let maxX = 0,
    maxY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let i = (x + y * width) * 4;
      let val = pixels[i];

      if (val > 20) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (minX >= maxX || minY >= maxY) {
    console.log("Nothing detected");
    return;
  }

  let w = maxX - minX;
  let h = maxY - minY;

  let cropped = createImage(w, h);
  cropped.loadPixels();

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let srcI = (x + minX + (y + minY) * width) * 4;
      let dstI = (x + y * w) * 4;

      cropped.pixels[dstI] = pixels[srcI];
      cropped.pixels[dstI + 1] = pixels[srcI + 1];
      cropped.pixels[dstI + 2] = pixels[srcI + 2];
      cropped.pixels[dstI + 3] = 255;
    }
  }
  cropped.updatePixels();

  let padding = 30;
  let size = max(w, h) + padding;

  let square = createImage(size, size);
  square.loadPixels();

  // fill black
  for (let i = 0; i < square.pixels.length; i += 4) {
    square.pixels[i] = 0;
    square.pixels[i + 1] = 0;
    square.pixels[i + 2] = 0;
    square.pixels[i + 3] = 255;
  }
  square.updatePixels();

  let offsetX = floor((size - w) / 2);
  let offsetY = floor((size - h) / 2);

  square.copy(cropped, 0, 0, w, h, offsetX, offsetY, w, h);

  let finalImg = createImage(28, 28);
  finalImg.copy(square, 0, 0, size, size, 0, 0, 28, 28);

  finalImg.loadPixels();

  let screenPixels = [];

  for (let i = 0; i < finalImg.pixels.length; i += 4) {
    screenPixels.push(finalImg.pixels[i]);
  }

  screenPixels = greyToMany(screenPixels, initChannels);

  let screenChannels = [];
  screenPixels.forEach((pixels) => {
    screenChannels.push(new CNNChannel(pixels, 28, 28));
  });

  let screenCNNImg = new CNNImage(
    screenChannels,
    CNN.imgWidth,
    CNN.imgHeight,
    screenChannels.length,
  );

  let prediction = brain.predict(screenCNNImg);
  let sum = prediction.reduce((a, b) => a + b);
  prediction = prediction.map((p) => p / sum);

  let highestInd = prediction.indexOf(Math.max(...prediction));

  document.getElementById("computerGuess").innerHTML =
    `Guessed: ${highestInd} <br>
     ${prediction.map((p, i) => `${i}: ${(p * 100).toFixed(2)}%`).join("<br>")}`;
}

//function guessFromScreen() {
//let screenImg = createImage(28, 28);
//screenImg.copy(get(), 0, 0, width, height, 0, 0, 28, 28);
//screenImg.loadPixels();
//let screenPixels = [];
//for (let i = 0; i < screenImg.pixels.length; i += 4) {
//screenPixels.push(screenImg.pixels[i]);
//}
//screenPixels = greyToMany(screenPixels, initChannels);

//let screenChannels = [];
//screenPixels.forEach((pixels) => {
//let channel = new CNNChannel(pixels, 28, 28);
//screenChannels.push(channel);
//});
//let screenCNNImg = new CNNImage(
//screenChannels,
//CNN.imgWidth,
//CNN.imgHeight,
//screenChannels.length,
//);
//let prediction = brain.predict(screenCNNImg);
//let highestInd = prediction.indexOf(Math.max(...prediction));
//let sum = prediction.reduce((a, b) => a + b);
//prediction.forEach((p, i) => {
//prediction[i] = p / sum;
//console.log(i + ": " + prediction[i].toFixed(2) * 100 + "%");
//});
//console.log("-----------------------------------------------");
//console.log("-----------------------------------------------");
//console.log("-----------------------------------------------");
//document.getElementById("computerGuess").innerHTML =
//`Guessed: ${highestInd} <br>
//Probabilities: ${prediction.map((p, i) => `${i}: ${prediction[i].toFixed(2) * 100}%`).join("<br>")}
//`;
//}
