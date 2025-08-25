const p5 = require('p5');

new p5(p => {
  p.setup = () => {
    p.createCanvas(800, 600, p.WEBGL);
    p.noStroke();
  };
  p.draw = () => {
    p.background(0);
    p.rotateX(p.frameCount * 0.01);
    p.rotateY(p.frameCount * 0.01);
    if (p.mouseIsPressed) p.scale(1.1);
    p.box(100);
  };
});
