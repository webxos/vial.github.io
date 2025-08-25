const p5 = require('p5');

new p5(p => {
  p.setup = () => {
    p.createCanvas(800, 600, p.WEBGL);
  };
  p.draw = () => {
    p.background(0);
    p.rotateX(p.frameCount * 0.01);
    p.rotateY(p.frameCount * 0.01);
    p.box(100);
  };
});
