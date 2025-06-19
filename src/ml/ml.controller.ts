import { Controller, Post, Body, UploadedFile, UseInterceptors, BadRequestException, Get, Res } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { MlService } from './ml.service';
import { Express, Response } from 'express';

@Controller('ml')
export class MlController {
  constructor(private readonly mlService: MlService) {}

  @Post('predict')
  async predict(@Body('skills') skills: string) {
    return this.mlService.predict(skills);
  }

  @Post('upload-csv')
  @UseInterceptors(FileInterceptor('file'))
  async uploadCsv(@UploadedFile() file: Express.Multer.File) {
    if (!file) {
      return { message: 'No file uploaded', error: 'Bad Request', statusCode: 400 };
    }
    return this.mlService.importCsv(file);
  }

  @Post('add-title')
  async addTitle(@Body() body: { Title: string; Skills: string }) {
    if (!body.Title || !body.Skills) {
      return { message: 'Title and Skills are required', error: 'Bad Request', statusCode: 400 };
    }
    return this.mlService.addTitle(body);
  }

  @Get('export-csv')
  async exportCsvFile() {
    return this.mlService.exportTitlesToCsvFile();
  }
  @Get('train-now')
async triggerTrainingNow() {
  await this.mlService.handleTrainingJob(); // call the cron method directly
  return { message: 'Training job triggered manually' };
}
}
