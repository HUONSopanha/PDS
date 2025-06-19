import { Injectable, Logger } from '@nestjs/common';
import { spawn } from 'child_process';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Title } from './entities/title.entity';
import * as csvParser from 'csv-parser';
import { Readable } from 'stream';
import { Express } from 'express'; // <-- Add this import
import * as fs from 'fs';
import * as path from 'path';
// Removed incorrect import for Multer's File type
import { Cron, CronExpression } from '@nestjs/schedule';

@Injectable()
export class MlService {
  private readonly logger = new Logger(MlService.name);

  constructor(
    @InjectRepository(Title)
    private readonly titleRepository: Repository<Title>,
  ) {}

  async predict(userSkills: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const env = {
        ...process.env,
        CUDA_VISIBLE_DEVICES: '',
        TF_CPP_MIN_LOG_LEVEL: '3',
      };

      const py = spawn('python3', ['src/predict.py', userSkills], { env });

      let output = '';
      let error = '';

      py.stdout.on('data', (data) => {
        output += data.toString();
      });

      py.stderr.on('data', (data) => {
        const str = data.toString();
        const knownWarnings = [
          'Unable to register cuDNN',
          'Unable to register cuBLAS',
          'Unable to register cuFFT',
          'computation placer already registered',
          'written to STDERR',
        ];
        const isNoise = knownWarnings.some((pattern) => str.includes(pattern));
        if (!isNoise) {
          error += str;
        } else {
          this.logger.warn(`Filtered Python warning: ${str.trim()}`);
        }
      });

      py.on('close', (code) => {
        if (code !== 0 || error.trim()) {
          this.logger.error(`Python exited with code ${code}, stderr: ${error}`);
          return reject(`Python exited with code ${code}, stderr: ${error || 'Unknown error'}`);
        }

        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (parseErr) {
          this.logger.error('Failed to parse Python output', parseErr);
          reject(`Failed to parse output: ${output}`);
        }
      });
    });
  }

  async importCsv(file: Express.Multer.File): Promise<{ message: string; count: number }> {
    const results: Partial<Title>[] = [];
    const stream = Readable.from(file.buffer);

    return new Promise((resolve, reject) => {
      stream
        .pipe(csvParser())
        .on('data', (data) => {
          // Log or validate data here
          const title = data['Title']?.trim();
          const skills = data['Skills']?.trim();

          if (!title || !skills) {
            return reject(new Error('Missing required fields "Title" or "Skills" in CSV'));
          }

          results.push({ Title: title, Skills: skills });
        })
        .on('end', async () => {
          try {
            await this.titleRepository.save(results);
            resolve({ message: 'CSV data imported successfully', count: results.length });
          } catch (err) {
            reject(new Error('Failed to save CSV data: ' + err.message));
          }
        })
        .on('error', (err) => reject(new Error('CSV parsing error: ' + err.message)));
    });
  }

  async addTitle(data: { Title: string; Skills: string }): Promise<Title> {
    const newTitle = this.titleRepository.create(data);
    return this.titleRepository.save(newTitle);
  }

  async exportTitlesToCsvFile(): Promise<{ path: string }> {
    const titles = await this.titleRepository.find();
    const header = 'Title,Skills\n';
    const rows = titles.map(t => `"${t.Title.replace(/"/g, '""')}","${t.Skills.replace(/"/g, '""')}"`).join('\n');
    const csvContent = header + rows + '\n';

    const dir = '/home/sopanha/Class/I3/Semister_2/PDS/Project/backend/src/update-ml';
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    const filePath = path.join(dir, 'titles.csv');
    fs.writeFileSync(filePath, csvContent, 'utf8');
    return { path: filePath };
  }
  
  @Cron(CronExpression.EVERY_DAY_AT_MIDNIGHT)
async handleTrainingJob() {
  this.logger.log('Starting daily ML model training job at 12 AM');

  try {
    // Step 1: Export Titles to CSV file
    const exportResult = await this.exportTitlesToCsvFile();
    const exportPath = exportResult.path;
    this.logger.log(`Exported Titles CSV to ${exportPath}`);

    // Step 2: Run Python training script with CSV file path as argument
    const env = {
      ...process.env,
      CUDA_VISIBLE_DEVICES: '', // Disable GPU if needed
      TF_CPP_MIN_LOG_LEVEL: '3',
    };

    const scriptPath = path.resolve('/home/sopanha/Class/I3/Semister_2/PDS/Project/backend/src/update-ml/train.py');

    // Pass export CSV path as argument to Python script
    const trainProcess = spawn('python3', [scriptPath, exportPath], { env, shell: true });

    trainProcess.stdout.on('data', (data) => {
      this.logger.log(`Training stdout: ${data.toString()}`);
    });

    trainProcess.stderr.on('data', (data) => {
      const str = data.toString();
      const knownWarnings = [
        'Unable to register cuDNN',
        'Unable to register cuBLAS',
        'Unable to register cuFFT',
        'computation placer already registered',
        'written to STDERR',
      ];
      const isNoise = knownWarnings.some((pattern) => str.includes(pattern));
      if (!isNoise) {
        this.logger.error(`Training stderr: ${str}`);
      } else {
        this.logger.warn(`Filtered Training warning: ${str.trim()}`);
      }
    });

    trainProcess.on('close', (code) => {
      if (code === 0) {
        this.logger.log('Training script finished successfully');
      } else {
        this.logger.error(`Training script exited with code ${code}`);
      }
    });

  } catch (error) {
    this.logger.error(`Failed to export CSV or run training: ${error.message}`);
  }
}
}

