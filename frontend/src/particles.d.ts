declare module 'particles.js' {
    interface Particles {
      load(tagId: string, pathConfigJson: string, callback: () => void): void;
    }
  
    const particlesJS: Particles;
  
    export default particlesJS;
  }