// src/dialogflow/dialogflow.controller.ts
import { Controller, Get, Query } from '@nestjs/common';
import { DialogflowService } from './dialogflow.service';

@Controller('dialogflow')
export class DialogflowController {
  constructor(private readonly dfService: DialogflowService) {}

  @Get()
  async query(@Query('q') q: string) {
    const response = await this.dfService.detectIntent(q, 'unique-session-id');
    return { reply: response };
  }
}
