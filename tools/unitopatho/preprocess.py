import os
import sys
import zipfile
import shutil
import pandas as pd
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class UniToPathoPreprocessor:
    
    def __init__(self, input_dir: str = "./unitopatho_raw", output_dir: str = "./unitopatho"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.temp_extract = self.output_dir / "temp_extract"
        
    def extract_zips(self):
        logger.info("Extracting ZIP files...")
        self.temp_extract.mkdir(exist_ok=True)
        
        zip_files = list(self.input_dir.glob("*.zip"))
        if not zip_files:
            logger.error(f"No ZIP files found in {self.input_dir}")
            return False
            
        for zip_path in zip_files:
            logger.info(f"Extracting {zip_path.name}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.temp_extract)
            except Exception as e:
                logger.error(f"Failed to extract {zip_path.name}: {e}")
                return False
                
        return True
    
    def explore_structure(self):
        logger.info("Exploring dataset structure...")
        
        found = {
            '800_dir': None,
            '7000_dir': None,
            'csv_files': []
        }
        
        for root, dirs, files in os.walk(self.temp_extract):
            root_path = Path(root)
            
            if '800' in dirs:
                found['800_dir'] = root_path / '800'
                logger.info(f"Found 800 directory: {found['800_dir']}")
            elif root_path.name == '800':
                found['800_dir'] = root_path
                logger.info(f"Found 800 directory: {found['800_dir']}")
                
            if '7000' in dirs:
                found['7000_dir'] = root_path / '7000'
                logger.info(f"Found 7000 directory: {found['7000_dir']}")
            elif root_path.name == '7000':
                found['7000_dir'] = root_path
                logger.info(f"Found 7000 directory: {found['7000_dir']}")
            
            for file in files:
                if file.endswith('.csv'):
                    csv_path = root_path / file
                    found['csv_files'].append(csv_path)
                    logger.info(f"Found CSV: {csv_path}")
                    
        return found
    
    def analyze_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        logger.info(f"CSV {csv_path.name}: {len(df)} rows")
        logger.info(f"Columns: {list(df.columns)}")
        
        if 'top_label_name' in df.columns:
            label_counts = df['top_label_name'].value_counts()
            logger.info(f"Label distribution: {dict(label_counts)}")
        elif 'label' in df.columns:
            label_counts = df['label'].value_counts()
            logger.info(f"Label distribution: {dict(label_counts)}")
            
        return df
    
    def organize_images(self, structure):
        if not structure['800_dir']:
            logger.error("800 directory not found!")
            return False
            
        source_800 = structure['800_dir']
        target_800 = self.output_dir / '800'
        
        classes = ['HP', 'NORM', 'TA.HG', 'TA.LG', 'TVA.HG', 'TVA.LG']
        
        for class_name in classes:
            (target_800 / class_name).mkdir(exist_ok=True, parents=True)
            
        csv_files_800 = [f for f in structure['csv_files'] if '800' in str(f.parent)]
        
        if csv_files_800:
            logger.info("Using CSV to organize images...")
            
            for csv_path in csv_files_800:
                df = pd.read_csv(csv_path)
                
                if 'image_id' in df.columns and 'top_label_name' in df.columns:
                    for _, row in df.iterrows():
                        image_id = row['image_id']
                        label = row['top_label_name']
                        
                        source_paths = [
                            source_800 / label / image_id,
                            source_800 / image_id,
                        ]
                        
                        for source_path in source_paths:
                            if source_path.exists():
                                target_path = target_800 / label / image_id
                                shutil.copy2(source_path, target_path)
                                break
                                
                csv_name = csv_path.name
                target_csv = self.output_dir / csv_name
                shutil.copy2(csv_path, target_csv)
                logger.info(f"Copied {csv_name} to output directory")
                
        else:
            logger.info("No CSV files found for 800 subset, checking existing structure...")
            
            for class_name in classes:
                source_class = source_800 / class_name
                if source_class.exists() and source_class.is_dir():
                    target_class = target_800 / class_name
                    for img_file in source_class.glob('*'):
                        if img_file.is_file():
                            shutil.copy2(img_file, target_class / img_file.name)
                            
            image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
            for img_file in source_800.glob('*'):
                if img_file.is_file() and img_file.suffix.lower() in image_extensions:
                    for class_name in classes:
                        if class_name in img_file.stem.upper():
                            target_path = target_800 / class_name / img_file.name
                            shutil.copy2(img_file, target_path)
                            break
                            
        for csv_path in structure['csv_files']:
            if 'train' in csv_path.name.lower() or 'test' in csv_path.name.lower():
                target_csv = self.output_dir / csv_path.name
                if not target_csv.exists():
                    shutil.copy2(csv_path, target_csv)
                    logger.info(f"Copied {csv_path.name} to output directory")
                    
        return True
    
    def validate_output(self):
        logger.info("Validating output structure...")
        
        subset_800 = self.output_dir / '800'
        if not subset_800.exists():
            logger.error("800 directory missing in output!")
            return False
            
        classes = ['HP', 'NORM', 'TA.HG', 'TA.LG', 'TVA.HG', 'TVA.LG']
        total_images = 0
        
        for class_name in classes:
            class_dir = subset_800 / class_name
            if class_dir.exists():
                num_images = len(list(class_dir.glob('*')))
                total_images += num_images
                logger.info(f"Class {class_name}: {num_images} images")
                
        logger.info(f"Total images: {total_images}")
        
        train_csv = self.output_dir / 'train.csv'
        test_csv = self.output_dir / 'test.csv'
        
        if train_csv.exists():
            train_df = pd.read_csv(train_csv)
            logger.info(f"train.csv: {len(train_df)} samples")
        else:
            logger.warning("train.csv not found")
            
        if test_csv.exists():
            test_df = pd.read_csv(test_csv)
            logger.info(f"test.csv: {len(test_df)} samples")
        else:
            logger.warning("test.csv not found")
            
        metadata = {
            'dataset': 'UniToPatho',
            'subset': '800',
            'num_classes': 6,
            'classes': classes,
            'total_images': total_images,
            'image_size': '1812x1812'
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info("Created metadata.json")
        
        return total_images > 0
    
    def cleanup(self):
        if self.temp_extract.exists():
            shutil.rmtree(self.temp_extract)
            logger.info("Cleaned up temporary files")
    
    def run(self):
        logger.info("=" * 60)
        logger.info("UniToPatho Preprocessing")
        logger.info("=" * 60)
        
        try:
            if not self.extract_zips():
                return False
                
            structure = self.explore_structure()
            
            if structure['csv_files']:
                for csv_path in structure['csv_files']:
                    self.analyze_csv(csv_path)
                    
            if not self.organize_images(structure):
                return False
                
            valid = self.validate_output()
            
            self.cleanup()
            
            if valid:
                logger.info("=" * 60)
                logger.info("✓ Preprocessing complete!")
                logger.info(f"✓ Output: {self.output_dir.absolute()}")
                logger.info("=" * 60)
                return True
            else:
                logger.error("Validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess UniToPatho dataset")
    parser.add_argument("--input-dir", default="./unitopatho_raw", help="Input directory with ZIP files")
    parser.add_argument("--output-dir", default="./unitopatho", help="Output directory")
    
    args = parser.parse_args()
    
    preprocessor = UniToPathoPreprocessor(input_dir=args.input_dir, output_dir=args.output_dir)
    success = preprocessor.run()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()