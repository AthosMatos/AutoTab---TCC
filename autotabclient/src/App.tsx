import toast, { Toaster } from 'react-hot-toast';
import GeneralContext from './contexts/general_context';
import MainPage from './Pages/Main';
import styled from 'styled-components';

//full screen
const MainContainer = styled.div`
  height: 100vh;
  background-color: #1b1b1b;
`

function App() {

  return (
    <GeneralContext>
      <MainContainer>

        <Toaster
          position="top-right"
          reverseOrder={false}
        />
        <MainPage />
      </MainContainer>

    </GeneralContext>
  );
}

export default App;
