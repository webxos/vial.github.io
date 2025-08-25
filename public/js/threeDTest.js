const { createCanvas } = require('canvas');

class ThreeDTest {
  runTest() {
    const canvas = createCanvas(800, 600);
    console.log('3D Test Canvas Created:', canvas.width, canvas.height);
    return { status: 'success' };
  }
}

module.exports = new ThreeDTest();
