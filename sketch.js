let x_vals = [];
let y_vals = [];

// y = mx + b;
// 1. data set
// 2. predict
// 3. loss
// 4. optimizer
// 5. learningRate

const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);

let m, b;

function setup() {
    createCanvas(400, 400);
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function mousePressed() {
    const x = map(mouseX, 0, width, 0, 1);
    const y = map(mouseY, height, 0, 0, 1);
    x_vals.push(x);
    y_vals.push(y);
}

function loss(pred, labels) {
    // mean() => tinh trung binh
    return pred.sub(labels).square().mean();
}

function predict(x) {
    // compute ys
    const xs = tf.tensor1d(x);
    return m.mul(xs).add(b);
}

function draw() {
    tf.tidy(() => {
        if (x_vals.length > 0) {
            const ys = tf.tensor1d(y_vals);
            optimizer.minimize(() => loss(predict(x_vals), ys));
        }
    });
    background(0);
    stroke(255);
    strokeWeight(5);
    for (let i = 0; i < x_vals.length; i++) {
        const x = map(x_vals[i], 0, 1, 0, width);
        const y = map(y_vals[i], 0, 1, height, 0);
        point(x, y);
    }

    const lineX = [0, 1];
    tf.tidy(() => {
        const ys = predict(lineX).dataSync();
        const x0 = map(lineX[0], 0, 1, 0, width);
        const x1 = map(lineX[1], 0, 1, 0, width);
        const y0 = map(ys[0], 0, 1, height, 0);
        const y1 = map(ys[1], 0, 1, height, 0);
        stroke(255);
        strokeWeight(2);
        line(x0, y0, x1, y1);
    });
}