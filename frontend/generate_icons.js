import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

const publicDir = '/Users/akshleenkaur/Desktop/CrypticCoders/Cryptic-Coders/echolocator/frontend/public';
const iconsDir = path.join(publicDir, 'icons');

const sizes = [192, 512];

async function generateIcons() {
  const iconSvg = fs.readFileSync(path.join(iconsDir, 'icon.svg'));
  const maskableSvg = fs.readFileSync(path.join(iconsDir, 'icon-maskable.svg'));

  for (const size of sizes) {
    // Regular icons
    await sharp(iconSvg)
      .resize(size, size)
      .png()
      .toFile(path.join(iconsDir, `icon-${size}.png`));
    console.log(`Generated icon-${size}.png`);

    // Maskable icons
    await sharp(maskableSvg)
      .resize(size, size)
      .png()
      .toFile(path.join(iconsDir, `icon-maskable-${size}.png`));
    console.log(`Generated icon-maskable-${size}.png`);
  }
}

generateIcons().catch(console.error);
