import React from 'react';
import ReactDOM from 'react-dom';
import Home from './modules/shared/Home.jsx';
import { HashRouter as Router, Route } from 'react-router-dom'
const rootRoute = () => (
  <Router>
    <Route exact path='/' component={Home}/>
  </Router>
)
ReactDOM.render(React.createElement(rootRoute), document.getElementById('root'));
