import { Module } from '@nestjs/common';
import { OpenAIService } from './openai.service';
import { SkillsController } from './openai.controller';

@Module({
  providers: [OpenAIService],
  exports: [OpenAIService],
  controllers: [SkillsController],
})
export class OpenAIModule {}
