// src/dialogflow/dialogflow.service.ts
import { Injectable } from '@nestjs/common';
import { SessionsClient } from '@google-cloud/dialogflow';
import * as path from 'path';
import * as fs from 'fs';

@Injectable()
export class DialogflowService {
  private sessionClient: SessionsClient;
  private projectId: string;

  constructor() {
    const keyPath = '/home/sopanha/Downloads/potent-retina-462908-g9-596ed7f50718.json';
    this.sessionClient = new SessionsClient({ keyFilename: keyPath });
    this.projectId = JSON.parse(fs.readFileSync(keyPath, 'utf8')).project_id;
  }

  async detectIntent(text: string, sessionId: string) {
    const sessionPath = this.sessionClient.projectAgentSessionPath(this.projectId, sessionId);

    const request = {
      session: sessionPath,
      queryInput: {
        text: {
          text,
          languageCode: 'en-US',
        },
      },
    };

    const responses = await this.sessionClient.detectIntent(request);
    const result = responses[0].queryResult;
    return result ? result.fulfillmentText : null;
  }
}
