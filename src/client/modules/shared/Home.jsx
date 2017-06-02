import React from 'react';
import TextField from 'material-ui/TextField'
import RaisedButton from 'material-ui/RaisedButton'
import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider'
export default class Home extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      question1: '',
      question2: ''
    }
  }
  render() {

    return (
    <MuiThemeProvider>
     <div style={{textAlign: 'center'}}>
       <TextField hintText="First Question"
          onChange={this._changeQuestion1.bind(this)} value={this.state.question1} />
       <TextField hintText="Second Question"
         onChange={this._changeQuestion2.bind(this)} value={this.state.question2}/>
       <br></br>
       <RaisedButton primary={true} onClick={this._submitQuestions.bind(this)}>Submit</RaisedButton>

      </div>
      </MuiThemeProvider>
    )
  }
  _changeQuestion2(event,str) {
    this.setState({question2:str})
  }
  _changeQuestion1(event,str) {
    this.setState({question1:str})
  }
  _submitQuestions(event) {

  }
}
