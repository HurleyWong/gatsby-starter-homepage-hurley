import React from 'react';
import { Row, Col } from 'antd';
import ProgressBar from '../../Progress';

const SkillsProgress = () => (
  <div>
    <h2>My Skills</h2>
    <Row gutter={[20, 20]}>
      <Col xs={24} sm={24} md={12}>

        <ProgressBar
          percent={70}
          text="Java"
        />
        <ProgressBar
          percent={60}
          text="Python"
        />
        <ProgressBar
          percent={40}
          text="Kotlin"
        />
        <ProgressBar
          percent={40}
          text="Solidity"
        />
      </Col>
      <Col xs={24} sm={24} md={12}>
        <ProgressBar
          percent={70}
          text="Android"
        />
        <ProgressBar
          percent={50}
          text="JavaScript"
        />
        <ProgressBar
          percent={40}
          text="Flutter"
        />
        <ProgressBar
          percent={30}
          text="Swift"
        />
      </Col>
    </Row>
  </div>
);

export default SkillsProgress;
