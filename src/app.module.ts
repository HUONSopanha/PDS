import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { OpenAIModule } from './skill/openai.module';
import { MlController } from './ml/ml.controller';
import { MlService } from './ml/ml.service';
import { TypeOrmModule } from '@nestjs/typeorm';
import { Title } from './ml/entities/title.entity'; // Adjust the import path as needed

@Module({
  imports: [
    ConfigModule.forRoot(),
    OpenAIModule,
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: process.env.DB_HOST,
      port: parseInt(process.env.DB_PORT || '5432', 10),
      username: process.env.DB_USERNAME,
      password: process.env.DB_PASSWORD,
      database: process.env.DB_NAME,
      synchronize: true,
      autoLoadEntities: true,
    }),
    TypeOrmModule.forFeature([Title]),
  ],
  controllers: [MlController],
  providers: [MlService],
})
export class AppModule {}
