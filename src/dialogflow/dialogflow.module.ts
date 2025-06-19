import { Module } from '@nestjs/common';
import { DialogflowService } from './dialogflow.service';
import { DialogflowController } from './dialogflow.controller';

@Module({
  providers: [DialogflowService],
  controllers: [DialogflowController],
})
export class DialogflowModule {}
